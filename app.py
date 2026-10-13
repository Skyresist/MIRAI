"""MIRAI text/voice CLI. Run python app.py --help for all options."""
import argparse
import logging
from pathlib import Path
import sys

from assistant import Assistant, build_context
from config import Settings
from embeddings import EMBEDDING_MODELS
from glossary import Glossary
from retriever import PaperRAGRetriever, Retriever
from secret import patch_response
from utils.pdf_loader import load_pdfs
from visitor import IdleTimeout, VisitorSession


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def parser():
    defaults = Settings()
    result = argparse.ArgumentParser(description='MIRAI: local BINUS ASO PDF assistant with per-visitor memory.')
    result.add_argument('--question', '-q', help='Ask one question and exit.')
    result.add_argument('--pdf-dir', type=Path, default=defaults.pdf_dir)
    result.add_argument('--state-dir', type=Path, default=defaults.state_dir)
    result.add_argument('--session', default='default', help='Visitor label; every launch still starts fresh.')
    result.add_argument('--idle-timeout', type=positive, default=300,
                        help='Clear the visitor conversation after this many idle seconds; stay running (default 300).')
    result.add_argument('--model', default=defaults.model)
    result.add_argument('--low-memory', action='store_true',
                        help='Use quantized semantic PDF search, a smaller context and shorter replies for an 8 GB Jetson.')
    result.add_argument('--embedding-backend', choices=['dense', 'onnx', 'tfidf'],
                        help='Default dense; low-memory mode uses quantized ONNX semantic search.')
    result.add_argument('--embedding-model', choices=sorted(EMBEDDING_MODELS), default=defaults.embedding_model,
                        help='Sentence encoder for PDF search (compare with scripts/eval_retrieval.py).')
    result.add_argument('--embedding-device', choices=['cpu', 'cuda'], default='cpu')
    result.add_argument('--top-k', type=positive, help='PDF matches; default 4, or 2 in low-memory mode.')
    result.add_argument('--threshold', type=float, help="Minimum PDF match score; default depends on the embedding model.")
    result.add_argument('--max-tokens', type=positive, help='Reply tokens; default 192, or 160 in low-memory mode.')
    result.add_argument('--num-ctx', type=positive,
                        help='Ollama context tokens; default 4096, or 2048 in low-memory mode.')
    result.add_argument('--think', action=argparse.BooleanOptionalAction, default=False)
    result.add_argument('--glossary', type=Path, default=defaults.glossary_path)
    result.add_argument('--paperrag-index', type=Path, help='Use an existing trusted PaperRAG index instead of the built-in PDF loader.')
    result.add_argument('--retrieve-only', action='store_true', help='Show PDF matches without calling Ollama.')
    result.add_argument('--audio', type=Path, help='Transcribe an audio file with faster-whisper base.')
    result.add_argument('--language', choices=['auto', 'id', 'en'], default='auto')
    result.add_argument('--speech-device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    result.add_argument('--whisper-model', default='base', help='Model name or path to an already downloaded CTranslate2 model.')
    result.add_argument('--speak', type=Path, help='Write a Piper WAV answer; no sound device required.')
    result.add_argument('--play', action=argparse.BooleanOptionalAction, default=False,
                        help='Speak answers through the default system sound output. The start script enables this.')
    result.add_argument('--voices-dir', type=Path, default=defaults.voices_dir)
    result.add_argument('--verbose', action='store_true')
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.threshold is not None and not 0 <= arguments.threshold <= 1:
        parser().error('--threshold must be between 0 and 1')
    if arguments.audio and arguments.question:
        parser().error('--audio and --question cannot be combined')
    if arguments.low_memory and arguments.paperrag_index:
        parser().error('--low-memory uses built-in PDF search; omit --paperrag-index')
    if arguments.low_memory and arguments.embedding_backend == 'dense':
        parser().error('--low-memory uses ONNX semantic search; omit --embedding-backend dense')
    logging.basicConfig(level=logging.INFO if arguments.verbose else logging.WARNING, format='%(levelname)s: %(message)s')
    settings = build_settings(arguments)
    glossary = Glossary(settings.glossary_path)
    retriever = build_retriever(settings, arguments.paperrag_index)
    with VisitorSession(arguments.idle_timeout, settings.session) as visitor:
        assistant = Assistant(settings, retriever, glossary=glossary, memory=visitor.memory)
        return run_chat(arguments, settings, retriever, assistant, glossary, visitor)


def build_settings(arguments):
    """Shared by the terminal CLI and the touchscreen server."""
    defaults = Settings()
    settings = Settings(
        model=arguments.model, pdf_dir=arguments.pdf_dir, state_dir=arguments.state_dir,
        session=arguments.session, embedding_backend=arguments.embedding_backend or ('onnx' if arguments.low_memory else 'dense'),
        embedding_model=arguments.embedding_model,
        embedding_device=arguments.embedding_device, score_threshold=arguments.threshold,
        top_k=arguments.top_k or (2 if arguments.low_memory else defaults.top_k),
        max_tokens=arguments.max_tokens or (160 if arguments.low_memory else defaults.max_tokens),
        num_ctx=arguments.num_ctx or (2048 if arguments.low_memory else defaults.num_ctx), think=arguments.think,
        glossary_path=arguments.glossary, whisper_model=arguments.whisper_model,
        speech_device=arguments.speech_device, voices_dir=arguments.voices_dir,
    )
    if arguments.low_memory:
        settings.history_turns = 3
        settings.history_chars = 900
        settings.context_chars = 2200
        print(f'Low-memory mode: {settings.embedding_backend} PDF search, smaller recent-memory and evidence windows. '
              'Conversation memory lasts only for this visit.', file=sys.stderr)
    return settings


def build_retriever(settings, paperrag_index=None):
    cache_name = settings.embedding_model.rsplit('/', 1)[-1]
    retriever = (PaperRAGRetriever(paperrag_index, threshold=settings.score_threshold)
                 if paperrag_index else Retriever(
                     load_pdfs(settings.pdf_dir), cache_path=settings.state_dir / (
                         f'embeddings-onnx-{cache_name}.npz' if settings.embedding_backend == 'onnx' else f'embeddings-{cache_name}.npz'),
                     model_name=settings.embedding_model, backend=settings.embedding_backend,
                     device=settings.embedding_device, threshold=settings.score_threshold))
    if not retriever.has_data():
        print(f'No PDF knowledge loaded. Add university PDFs to {settings.pdf_dir}.', file=sys.stderr)
    return retriever


def run_chat(arguments, settings, retriever, assistant, glossary, visitor):
    language = None if arguments.language == 'auto' else arguments.language
    one_shot = arguments.question is not None or arguments.audio is not None
    question = arguments.question
    speech = None
    if arguments.audio or arguments.speak or arguments.play:
        from speech import Speech
        speech = Speech(settings, glossary)
    if arguments.audio:
        question, detected = speech.transcribe(arguments.audio, language=language)
        language = language or detected
        print('You (transcribed):', question)
    while True:
        if not one_shot:
            try:
                question = visitor.read()
            except IdleTimeout:
                was_active = visitor.active
                visitor.reset()
                if was_active:
                    print('\nBot: Conversation cleared after inactivity. Ready for a new visitor.')
                continue
            except (EOFError, KeyboardInterrupt):
                print('\nBot: Goodbye.')
                return 0
        if question is None or not question.strip():
            if one_shot:
                raise ValueError('No question or speech was detected.')
            print('Bot: Please enter a question.')
            continue
        command = visitor.command(question)
        if command == 'exit':
            print('Bot: Goodbye.')
            return 0
        if command == 'reset':
            print('Bot: Conversation cleared. Ready for a new visitor.')
        elif 'who is the number one idol' in question.casefold():
            print('Bot:', patch_response('K4N_7A'))
        elif arguments.retrieve_only:
            results = retriever.retrieve(glossary.expand_query(question), top_k=settings.top_k)
            print(build_context(results) or 'No relevant PDF evidence found.')
        else:
            answer = assistant.ask(question, language)
            visitor.after_answer(answer.language)
            print('Bot:', patch_response(answer.text))
            if answer.sources:
                print('Sources:')
                for index, source in enumerate(answer.sources, start=1):
                    location = f"p. {source['page']}" if source.get('page') else source.get('section', 'document')
                    print(f"  [{index}] {source['source']} ({location})")
            if arguments.speak or arguments.play:
                output = arguments.speak or settings.state_dir / 'last-answer.wav'
                try:
                    speech.synthesize(answer.text, output, answer.language)
                    print('Audio saved:', output)
                    if arguments.play:
                        speech.play(output)
                except Exception as error:
                    # A disconnected speaker or missing voice must not end a text conversation.
                    print(f'Speech unavailable: {error}', file=sys.stderr)
        if one_shot:
            return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'Error: {error}', file=sys.stderr)
        print('Check Ollama, downloaded models and --help; use --verbose for loading details.', file=sys.stderr)
        raise SystemExit(1)
