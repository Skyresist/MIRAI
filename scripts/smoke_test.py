"""Functional checks with temporary test PDFs; never add fictitious campus facts to data/."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant import Assistant
from config import Settings
from glossary import Glossary
from memory import Memory
from retriever import Retriever
from utils.pdf_loader import load_pdfs


def fixture_pdf(path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 50 700 Td (Apa saja UKM di BINUS ASO?) Tj 0 -20 Td (Dokumen uji: Unit Kegiatan Mahasiswa terdiri dari D'BASE dan AeroBASE.) Tj ET")
    page[NameObject('/Contents')] = writer._add_object(stream)
    writer.write(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Require a running Ollama and test generation/memory/tool calls.')
    parser.add_argument('--voice', action='store_true', help='Test both Piper voices and faster-whisper with downloaded weights.')
    parser.add_argument('--embedding-backend', choices=['dense', 'onnx', 'tfidf'], default='dense')
    parser.add_argument('--num-ctx', type=int, default=4096)
    args = parser.parse_args(argv)
    if args.num_ctx < 1:
        parser.error('--num-ctx must be positive')
    with tempfile.TemporaryDirectory(prefix='mirai-smoke-') as temporary:
        root = Path(temporary)
        fixture_pdf(root / 'test-only.pdf')
        settings = Settings(pdf_dir=root, state_dir=root / 'state', embedding_backend=args.embedding_backend,
                            num_ctx=args.num_ctx)
        documents = load_pdfs(root)
        cache = settings.state_dir / 'embeddings.npz'
        retriever = Retriever(documents, backend=args.embedding_backend, cache_path=cache)
        hits = retriever.retrieve(Glossary().expand_query('Apa saja UKM di BINUS ASO?'))
        assert hits and hits[0]['page'] == 1 and "D'BASE" in hits[0]['text'], hits
        if args.embedding_backend != 'tfidf':
            assert Retriever(documents, backend=args.embedding_backend, cache_path=cache).cache_hit
        print('PASS: PDF extraction, relevant retrieval, page provenance and embedding cache (when selected)', flush=True)
        memory = Memory()
        memory.append('Nama saya Raka', 'Halo Raka')
        assert memory.search('Raka')
        assert Memory().recent() == []
        print('PASS: visitor memory is searchable and a new visitor starts fresh', flush=True)
        if args.live:
            assistant = Assistant(settings, retriever, memory=memory)
            answer = assistant.ask('Apa saja UKM di BINUS ASO?')
            assert "D'BASE" in answer.text and 'AeroBASE' in answer.text, answer.text
            assert answer.sources[0]['page'] == 1
            print('PASS: Qwen3 university answer grounded in test PDF:', answer.text, flush=True)
            recalled = assistant.ask('Ingat nama saya?')
            assert 'Raka' in recalled.text, recalled.text
            print('PASS: Qwen3 remembers a name during the same visit', flush=True)
            fresh = Assistant(settings, retriever).ask('Ingat nama saya?')
            assert 'Raka' not in fresh.text, fresh.text
            print('PASS: Qwen3 does not remember the previous visitor', flush=True)
            older_settings = replace(settings, session='older-name-test')
            older_memory = Memory(session=older_settings.session)
            older_memory.append('Nama saya Raka', 'Halo Raka')
            for _ in range(older_settings.history_turns + 1):
                older_memory.append('Halo', 'Halo')
            assert not any('Raka' in turn['content'] for turn in older_memory.recent(
                older_settings.history_turns, older_settings.history_chars))
            old_name = Assistant(older_settings, retriever, memory=older_memory).ask('Ingat nama saya?')
            assert 'Raka' in old_name.text, old_name.text
            print('PASS: Qwen3 recalls a name outside the recent-history window', flush=True)
            calls = []

            def recall_memory(query: str) -> str:
                """Find a stored user's name. Search for Raka to retrieve the test record."""
                calls.append(query)
                return '{"name": "Raka"}'

            result = assistant.chat.answer('Use recall_memory with query Raka, then tell me the returned name.', '', [], tools={'recall_memory': recall_memory})
            assert calls and 'Raka' in result, (calls, result)
            print('PASS: real Ollama Python tool-call round trip', flush=True)
        if args.voice:
            from speech import Speech
            speech = Speech(settings, Glossary())
            for language, text in [('id', 'Apa saja UKM di BINUS ASO?'), ('en', 'What student clubs are available at BINUS ASO?')]:
                audio = speech.synthesize(text, root / f'{language}.wav', language)
                with wave.open(str(audio)) as wav:
                    assert wav.getnframes() > 100 and wav.getframerate() > 8000
                transcript, _ = speech.transcribe(audio, language)
                assert transcript.strip()
                assert 'BINUS ASO' in transcript, transcript
                if language == 'id':
                    assert 'UKM' in transcript, transcript
                print(f'PASS: {language} Piper WAV -> faster-whisper base: {transcript}', flush=True)
            silence = root / 'silence.wav'
            with wave.open(str(silence), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b'\x00' * 32000)
            try:
                speech.transcribe(silence, 'id')
            except ValueError:
                print('PASS: silence does not hallucinate a question', flush=True)
            else:
                raise AssertionError('Silence was transcribed as speech')


if __name__ == '__main__':
    main()
