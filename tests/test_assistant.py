from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from assistant import Assistant
from config import Settings
from glossary import Glossary
from llm import FALLBACK, UNIVERSITY_FALLBACK, OllamaChat, detect_language
from memory import Memory
from retriever import Retriever
from utils.pdf_loader import load_pdfs


def write_pdf(path: Path, lines: list[str]):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    content = DecodedStreamObject()
    commands = ['BT /F1 12 Tf 50 700 Td']
    for line in lines:
        line = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        commands.append(f'({line}) Tj 0 -20 Td')
    content.set_data((' '.join(commands) + ' ET').encode())
    page[NameObject('/Contents')] = writer._add_object(content)
    writer.write(path)


def test_real_pdf_to_lexical_retrieval(tmp_path):
    nested = tmp_path / 'handbook'
    nested.mkdir()
    write_pdf(nested / 'faq.PDF', ['Apa saja UKM di BINUS ASO?', "Dalam dokumen uji ini, UKM meliputi D'BASE dan AeroBASE."])
    docs = load_pdfs(tmp_path)
    assert docs[0]['source'] == 'handbook/faq.PDF'
    assert docs[0]['page'] == 1
    retriever = Retriever(docs, backend='tfidf')
    hits = retriever.retrieve(Glossary().expand_query('u k m binus asso'))
    assert hits and "D'BASE" in hits[0]['text']
    assert retriever.retrieve('zzzxxyyqq') == []


def test_corrupt_pdf_is_reported(tmp_path, caplog):
    (tmp_path / 'broken.pdf').write_bytes(b'not a PDF')
    assert load_pdfs(tmp_path) == []
    assert 'Cannot read PDF' in caplog.text


def test_empty_collection_never_downloads_model(monkeypatch):
    monkeypatch.setattr(Retriever, '_model', Mock(side_effect=AssertionError('must not load')))
    retriever = Retriever([])
    assert not retriever.has_data()
    assert retriever.retrieve('hello') == []


def test_embedding_cache_reuse_and_content_invalidation(tmp_path, monkeypatch):
    model = Mock()
    model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(Retriever, '_model', lambda self: model)
    docs = [{'source': 'faq.pdf', 'page': 1, 'text': 'A university document with enough text for retrieval.'}]
    path = tmp_path / 'index.npz'
    first = Retriever(docs, cache_path=path)
    assert not first.cache_hit
    second = Retriever(docs, cache_path=path)
    assert second.cache_hit
    assert model.encode.call_count == 1
    docs[0]['text'] += ' Changed fee.'
    assert not Retriever(docs, cache_path=path).cache_hit
    assert model.encode.call_count == 2


def test_corrupt_cache_rebuilds(tmp_path, monkeypatch):
    model = Mock()
    model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(Retriever, '_model', lambda self: model)
    path = tmp_path / 'broken.npz'
    path.write_bytes(b'invalid archive')
    result = Retriever([{'source': 'faq.pdf', 'text': 'A full sentence describing university facilities.'}], cache_path=path)
    assert not result.cache_hit
    with np.load(path, allow_pickle=False) as cache:
        assert cache['embeddings'].shape == (1, 2)


def test_memory_survives_restart_and_isolates_sessions(tmp_path):
    path = tmp_path / 'memory.sqlite3'
    Memory(path, 'alice').append('Nama saya Raka', 'Halo Raka')
    reopened = Memory(path, 'alice')
    assert reopened.recent()[0]['content'] == 'Nama saya Raka'
    assert reopened.search('Raka')
    other = Memory(path, 'bob')
    other.append('hello', 'hi')
    assert not other.search('Raka')
    reopened.clear()
    assert reopened.recent() == []
    assert other.recent()


def test_memory_prompt_budget_preserves_whole_turns(tmp_path):
    memory = Memory(tmp_path / 'memory.sqlite3')
    memory.append('older question', 'older answer')
    memory.append('q', 'a')
    assert memory.recent(max_chars=2) == [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]


def test_memory_search_escapes_sql_patterns(tmp_path):
    memory = Memory(tmp_path / 'memory.sqlite3')
    memory.append('25% discount', 'test')
    memory.append('plain', 'test')
    assert len(memory.search('%')) == 1
    assert memory.search("' OR 1=1 --") == []


def test_glossary_and_pronunciation_preserve_word_boundaries():
    glossary = Glossary()
    assert glossary.normalize('u k m di binus asso') == 'UKM di BINUS ASO'
    assert glossary.normalize('ukmarine and database') == 'ukmarine and database'
    assert 'Unit Kegiatan Mahasiswa' in glossary.expand_query('UKM')
    assert glossary.pronounce('UKM BINUS ASO', 'id') == 'u ka em Binus Aso'
    assert ',' not in glossary.initial_prompt  # A word list in the prompt gets echoed by Whisper.
    assert glossary.normalize_speech("UKM d'Binus ASO") == 'UKM di BINUS ASO'
    # How Whisper writes spoken club names and letter-by-letter abbreviations.
    assert glossary.normalize_speech('Apa itu the base?') == "Apa itu D'BASE?"
    assert glossary.normalize_speech('ikut di base dan aero beis') == "ikut D'BASE dan AeroBASE"
    assert glossary.normalize_speech('Apa itu tepekaes dan u ka em?') == 'Apa itu TPKS dan UKM?'
    # Spelled-out letters arrive with any punctuation between them.
    for heard in ('Apa itu Ha, em, be?', 'Apa itu ha-em-be?', 'Apa itu Haembe?', 'Apa itu H. M. B.?'):
        assert glossary.normalize_speech(heard).startswith('Apa itu HMB'), heard
    assert glossary.normalize_speech('Te, pe, ka, es itu apa?') == 'TPKS itu apa?'
    owners = {}
    for term in glossary.terms:  # Ignoring separators must not make two names collide.
        for alias in term['aliases']:
            assert owners.setdefault(glossary._key(alias), term['canonical']) == term['canonical'], alias
    # "di"/"the" before BASE CORE must not turn it into D'BASE.
    assert glossary.normalize_speech('ikut di base core') == 'ikut BASE CORE'


def test_language_detection_uses_words():
    assert detect_language('Apa fasilitas di kampus?') == 'Bahasa Indonesia'
    assert detect_language('Halo') == 'Bahasa Indonesia'
    assert detect_language('Terima kasih') == 'Bahasa Indonesia'
    assert detect_language('Tell me about Canada') == 'English'


def test_unsupported_university_answer_does_not_call_llm(tmp_path):
    chat = Mock()
    assistant = Assistant(Settings(state_dir=tmp_path), Retriever([], backend='tfidf'), chat=chat)
    answer = assistant.ask('What are the tuition fees?')
    assert answer.text == UNIVERSITY_FALLBACK['en']
    chat.answer.assert_not_called()
    assert assistant.memory.recent()[0]['content'] == 'What are the tuition fees?'


def test_restart_history_is_actually_sent_to_llm(tmp_path):
    settings = Settings(state_dir=tmp_path)
    chat = Mock()
    chat.answer.return_value = 'Hello Raka'
    memory = Memory(tmp_path / 'explicit-history.sqlite3')
    Assistant(settings, Retriever([]), chat=chat, memory=memory).ask('My name is Raka')
    second = Assistant(settings, Retriever([]), chat=chat, memory=Memory(memory.path))
    second.ask('Remember my name?')
    history = chat.answer.call_args.args[2]
    assert any(message['content'] == 'My name is Raka' for message in history)


def test_personal_name_recall_ignores_irrelevant_pdf_hits(tmp_path):
    retriever = Mock()
    retriever.retrieve.return_value = [{'source': 'faq.pdf', 'page': 1, 'text': 'University facts, not a personal name.'}]
    chat = Mock()
    chat.answer.return_value = 'Your name is Raka.'
    assistant = Assistant(Settings(state_dir=tmp_path), retriever, chat=chat)
    assistant.memory.append('My name is Raka', 'Hello Raka')
    answer = assistant.ask('Remember my name?')
    assert answer.text == 'Your name is Raka.'
    assert answer.sources == []
    retriever.retrieve.assert_not_called()


def test_old_self_introduction_is_available_with_short_history_and_after_rename(tmp_path):
    settings = Settings(state_dir=tmp_path, history_turns=2, history_chars=200)
    chat = Mock()
    chat.answer.return_value = 'Your name is Raka.'
    assistant = Assistant(settings, Retriever([]), chat=chat)
    assistant.memory.append('Nama saya Raka', 'Halo Raka')
    for _ in range(4):
        assistant.memory.append('Hello', 'Hi')
    assistant.ask('Remember my name?')
    history = chat.answer.call_args.args[2]
    assert any(turn['content'] == 'Nama saya Raka' for turn in history)
    assert sum(len(turn['content']) for turn in history) <= settings.history_chars
    assistant.memory.append('My name is Andi', 'Hi Andi')
    assert assistant.memory.latest_name_turn()[0]['content'] == 'My name is Andi'
    assert Memory(session='other').latest_name_turn() == []
    assistant.memory.clear()
    assert assistant.memory.latest_name_turn() == []


def test_university_name_still_uses_pdf_evidence(tmp_path):
    retriever = Mock()
    retriever.retrieve.return_value = [{'source': 'faq.pdf', 'page': 1, 'text': 'The university name is BINUS ASO.'}]
    chat = Mock()
    chat.answer.return_value = 'BINUS ASO [1].'
    answer = Assistant(Settings(state_dir=tmp_path), retriever, chat=chat).ask("What is the university's name?")
    assert answer.sources[0]['source'] == 'faq.pdf'
    retriever.retrieve.assert_called_once()


class RecordingClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return {'message': next(self.replies)}


def test_ollama_tool_result_is_returned_to_model(tmp_path):
    client = RecordingClient([
        {'role': 'assistant', 'tool_calls': [{'function': {'name': 'recall_memory', 'arguments': {'query': 'name'}}}]},
        {'role': 'assistant', 'content': 'Your name is Raka.'},
    ])
    settings = Settings(state_dir=tmp_path)
    chat = OllamaChat(settings, client)
    recall = Mock(return_value='Raka')
    assert chat.answer('Remember my name?', '', [], tools={'recall_memory': recall}) == 'Your name is Raka.'
    recall.assert_called_once_with(query='name')
    assert client.requests[1]['messages'][-1] == {'role': 'tool', 'tool_name': 'recall_memory', 'content': 'Raka'}
    assert client.requests[0]['model'] == 'qwen3:4b-instruct-2507-q4_K_M'
    assert client.requests[0]['think'] is False


def test_unknown_tool_is_never_executed(tmp_path):
    client = RecordingClient([
        {'role': 'assistant', 'tool_calls': [{'function': {'name': 'shell', 'arguments': {'query': 'rm -rf'}}}]},
        {'role': 'assistant', 'content': 'No such tool.'},
    ])
    chat = OllamaChat(Settings(state_dir=tmp_path), client)
    chat.answer('hello', '', [], tools={'recall_memory': lambda query: ''})
    assert 'Unknown tool' in client.requests[1]['messages'][-1]['content']


def test_tool_loop_has_a_hard_limit(tmp_path):
    call = {'role': 'assistant', 'tool_calls': [{'function': {'name': 'recall_memory', 'arguments': {'query': 'name'}}}]}
    client = RecordingClient([call] * 4)
    with pytest.raises(RuntimeError, match='tool-call limit'):
        OllamaChat(Settings(state_dir=tmp_path), client).answer('hello', '', [], tools={'recall_memory': lambda query: ''})
    assert len(client.requests) == 4
    assert client.requests[-1]['tools'] is None


def test_blank_question_is_rejected_without_memory_write(tmp_path):
    assistant = Assistant(Settings(state_dir=tmp_path), Retriever([]), chat=Mock())
    with pytest.raises(ValueError):
        assistant.ask('  ')
    assert assistant.memory.recent() == []
