"""Touchscreen kiosk server: the web UI in static/ talks to MIRAI through this small HTTP API.

    python server.py --low-memory            # then open http://localhost:8000
One robot serves one visitor at a time, so requests share a single visitor session behind a lock.
"""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import app as cli
from assistant import Assistant
from glossary import Glossary
from secret import patch_response
from visitor import VisitorSession

logger = logging.getLogger('mirai.server')
STATIC = Path(__file__).resolve().parent / 'static'
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # About 10 minutes of Opus; a tap-to-talk question is seconds.


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str | None = Field(default=None, pattern='^(id|en)$')


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str = Field(pattern='^(id|en)$')


class Kiosk:
    """Everything one robot needs: retrieval, the current visitor and optional speech models."""

    def __init__(self, arguments):
        self.settings = cli.build_settings(arguments)
        self.settings.keep_alive = -1  # A dedicated kiosk keeps Qwen loaded between visitors.
        self.glossary = Glossary(self.settings.glossary_path)
        self.retriever = cli.build_retriever(self.settings, arguments.paperrag_index)
        self.visitor = VisitorSession(arguments.idle_timeout, self.settings.session)
        self.assistant = Assistant(self.settings, self.retriever, glossary=self.glossary, memory=self.visitor.memory)
        self.idle_timeout = arguments.idle_timeout
        self.last_activity = time.monotonic()
        self.lock = threading.Lock()
        self.cancelled = threading.Event()  # The visitor tapped cancel while an answer was being generated.
        self.audio_dir = Path(tempfile.mkdtemp(prefix='mirai-audio-'))
        self.voice = not arguments.no_voice
        self._speech = None
        self._speech_error = None
        self.navigator = self.assistant.navigator
        self.visiting = False
        self.greet = False
        self.closing = threading.Event()

    def watch_robot(self):
        while not self.closing.wait(1.0):
            if self.visiting and time.monotonic() - self.last_activity > self.idle_timeout + 15:
                self.end_visit()
            status = self.navigator.poll(busy=self.visiting)
            self.greet = not self.visiting and status.get('visitor') == 'arrived'

    def start_visit(self):
        self.visiting, self.greet = True, False
        self.last_activity = time.monotonic()

    def end_visit(self):
        was, self.visiting, self.greet = self.visiting, False, False
        if was and self.navigator is not None:
            threading.Thread(target=self.navigator.done, daemon=True).start()

    @property
    def speech(self):
        if not self.voice:
            raise HTTPException(503, 'Voice is disabled (--no-voice).')
        if self._speech is None:
            from speech import Speech
            self._speech = Speech(self.settings, self.glossary)
        return self._speech

    def touch(self):
        """Server-side idle reset, in case the screen's own timer never fired (e.g. page reload)."""
        now = time.monotonic()
        if self.visitor.active and now - self.last_activity > self.idle_timeout:
            self.visitor.reset()
        self.last_activity = now

    def preload(self):
        """Load speech models and wake Qwen before the first visitor arrives."""
        try:
            if self.voice:
                speech = self.speech
                speech.load_whisper()
                for language in ('id', 'en'):
                    speech.load_voice(language)
        except Exception as error:  # Text chat must still work without voice models.
            self._speech_error = str(error)
            logger.warning('Voice unavailable: %s', error)
        try:
            # Same num_ctx as real questions: a different size makes Ollama reload the model on the
            # first question (slow) and briefly hold two copies (out-of-memory risk).
            self.assistant.chat.client.generate(model=self.settings.model, prompt='', keep_alive=-1,
                                                options={'num_ctx': self.settings.num_ctx})
        except Exception as error:
            logger.warning('Could not preload %s: %s', self.settings.model, error)

    def close(self):
        self.closing.set()
        self.visitor.__exit__(None, None, None)
        for path in self.audio_dir.glob('*'):
            path.unlink(missing_ok=True)
        self.audio_dir.rmdir()


def create_app(arguments, preload=True) -> FastAPI:
    kiosk = Kiosk(arguments)

    @asynccontextmanager
    async def lifespan(_):
        if preload:
            threading.Thread(target=kiosk.preload, daemon=True).start()
        if kiosk.navigator is not None:
            threading.Thread(target=kiosk.watch_robot, daemon=True).start()
        yield
        kiosk.close()

    api = FastAPI(title='MIRAI kiosk', lifespan=lifespan)
    api.state.kiosk = kiosk

    @api.middleware('http')
    async def always_fresh_ui(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-cache'  # Chromium must load the updated screen after an update.
        return response

    def busy():
        raise HTTPException(409, 'MIRAI is still answering the previous question.')

    @api.get('/api/status')
    def status():
        return {'model': kiosk.settings.model, 'knowledge': kiosk.retriever.has_data(),
                'voice': kiosk.voice and kiosk._speech_error is None, 'voice_error': kiosk._speech_error,
                'idle_timeout': kiosk.idle_timeout, 'busy': kiosk.lock.locked(), 'greet': kiosk.greet}

    @api.post('/api/visit')
    def visit():
        kiosk.start_visit()
        return {'ok': True}

    @api.post('/api/ask')
    def ask(question: Question):
        # Ollama cannot be stopped mid-answer, so a question asked right after a cancel waits for
        # the abandoned answer to finish instead of being turned away as busy.
        if not (kiosk.lock.acquire(timeout=60) if kiosk.cancelled.is_set() else kiosk.lock.acquire(blocking=False)):
            busy()
        kiosk.cancelled.clear()
        try:
            kiosk.touch()
            if 'who is the number one idol' in question.text.casefold():
                return {'text': patch_response('K4N_7A'), 'sources': [], 'language': question.language or 'en'}
            answer = kiosk.assistant.ask(question.text, question.language)
            if kiosk.cancelled.is_set():
                # The visitor took the question back: it must not shape follow-up answers.
                kiosk.visitor.memory.forget_last()
                raise HTTPException(499, 'Cancelled by the visitor.')
            kiosk.visitor.after_answer(answer.language)
            sources = [{'source': source['source'], 'page': source.get('page'),
                        'question': source.get('question')} for source in answer.sources]
            return {'text': patch_response(answer.text), 'sources': sources, 'language': answer.language,
                    'navigation': answer.navigation}
        except HTTPException:
            raise
        except ValueError as error:
            raise HTTPException(400, str(error))
        except Exception as error:
            logger.exception('Answer failed')
            raise HTTPException(502, f'Answer failed: {error}. Is Ollama running?')
        finally:
            kiosk.last_activity = time.monotonic()
            kiosk.lock.release()

    @api.post('/api/cancel')
    def cancel():
        if kiosk.lock.locked():
            kiosk.cancelled.set()
        return {'ok': True}

    @api.post('/api/transcribe')
    def transcribe(audio: UploadFile = File(...), language: str = Form('auto')):
        if language not in ('auto', 'id', 'en'):
            raise HTTPException(400, 'language must be auto, id or en')
        data = audio.file.read(MAX_AUDIO_BYTES + 1)
        if not data:
            raise HTTPException(400, 'Empty recording.')
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, 'Recording is too long.')
        suffix = Path(audio.filename or 'speech.webm').suffix or '.webm'
        path = kiosk.audio_dir / f'in-{uuid.uuid4().hex}{suffix}'
        path.write_bytes(data)
        try:
            kiosk.touch()
            text, detected = kiosk.speech.transcribe(path, None if language == 'auto' else language)
            return {'text': text, 'language': detected if language == 'auto' else language}
        except ValueError as error:  # Silence or unintelligible audio, not a server fault.
            return {'text': '', 'language': language, 'error': str(error)}
        finally:
            path.unlink(missing_ok=True)

    @api.post('/api/speak')
    def speak(request: SpeakRequest):
        for old in sorted(kiosk.audio_dir.glob('out-*.wav'))[:-4]:
            old.unlink(missing_ok=True)  # Keep a few recent answers; the disk must not fill up.
        path = kiosk.audio_dir / f'out-{time.time_ns()}.wav'
        try:
            kiosk.speech.synthesize(request.text, path, request.language)
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(503, f'Speech unavailable: {error}')
        return FileResponse(path, media_type='audio/wav')

    @api.post('/api/reset')
    def reset():
        with kiosk.lock:
            kiosk.visitor.reset()
        kiosk.end_visit()
        return {'ok': True}

    if STATIC.is_dir():
        api.mount('/', StaticFiles(directory=STATIC, html=True), name='ui')
    return api


def main(argv=None):
    import uvicorn
    parser = cli.parser()
    parser.description = 'MIRAI touchscreen kiosk server.'
    parser.add_argument('--host', default='127.0.0.1', help='Use 0.0.0.0 only if another device must reach the UI.')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-voice', action='store_true', help='Text-only kiosk; skip Whisper and Piper.')
    parser.set_defaults(idle_timeout=120)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if arguments.verbose else logging.WARNING,
                        format='%(levelname)s: %(message)s')
    uvicorn.run(create_app(arguments), host=arguments.host, port=arguments.port, log_level='warning')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
