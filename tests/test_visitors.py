import os
import pty
import select
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

import app
import visitor
from assistant import Assistant
from config import Settings
from llm import FALLBACK, OllamaChat
from memory import Memory
from retriever import Retriever
from test_assistant import RecordingClient


def test_default_assistants_start_fresh_even_with_an_old_database(tmp_path):
    Memory(tmp_path / 'memory.sqlite3').append('My name is Previous', 'Hello Previous')
    first = Assistant(Settings(state_dir=tmp_path), Retriever([]), chat=Mock())
    first.memory.append('My name is Mirai', 'Hello Mirai')
    assert first.memory.search('Mirai')
    second = Assistant(Settings(state_dir=tmp_path), Retriever([]), chat=Mock())
    assert second.memory.recent() == []
    assert second.memory.search('Previous') == []
    first.memory.close()
    second.memory.close()


def test_generation_error_closes_memory_and_next_visit_starts_fresh():
    with pytest.raises(RuntimeError, match='generation failed'):
        with visitor.VisitorSession() as session:
            session.memory.append('Nama saya Mirai', 'Halo Mirai')
            raise RuntimeError('generation failed')
    assert session.memory._volatile is None
    with visitor.VisitorSession() as next_session:
        assert next_session.memory.recent() == []


@pytest.mark.parametrize('command', ['done', 'finish', 'finished', 'selesai', '/done',
                                   'DONE!', ' Finish. ', 'exit', 'bye'])
def test_finish_immediately_clears_visitor_without_closing_memory(command, monkeypatch):
    monkeypatch.setattr(visitor, 'timed_input', Mock(side_effect=AssertionError('No confirmation needed')))
    with visitor.VisitorSession() as session:
        session.memory.append('My name is Mirai', 'Hi Mirai')
        session.after_answer('id')
        assert 'done atau finish' in session.prompt
        assert session.command(command) == 'reset'
        assert session.memory.recent() == []
        assert session.memory.search('Mirai') == []
        assert session.memory.latest_name_turn() == []
        assert session.prompt == 'You: '
        session.memory.append('My name is Next', 'Hello Next')
        assert session.memory.search('Next')
        assert session.command('finish the explanation please') is None


@pytest.mark.parametrize('ending', ['exit', 'reset', 'done', 'finish', 'eof', 'interrupt', 'timeout', 'idle_without_visitor'])
def test_actual_cli_cleanup(tmp_path, monkeypatch, ending, capsys):
    seen = []

    def answer(chat, question, context, history, language, tools):
        seen.append(list(history))
        return 'Hello.'

    monkeypatch.setattr(OllamaChat, 'answer', answer)
    inputs = [] if ending == 'idle_without_visitor' else ['My name is Mirai']
    inputs += {'exit': ['/exit'], 'reset': ['/reset', 'Remember my name?', '/exit'],
               'done': ['done', 'Remember my name?', '/exit'],
               'finish': ['finish', 'Remember my name?', '/exit'],
               'eof': [EOFError()], 'interrupt': [KeyboardInterrupt()],
               'timeout': [visitor.IdleTimeout(), 'Remember my name?', '/exit'],
               'idle_without_visitor': [visitor.IdleTimeout(), visitor.IdleTimeout(), '/exit']}[ending]
    inputs = iter(inputs)

    def read(*args):
        item = next(inputs)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(visitor, 'timed_input', read)
    sessions = []
    original = visitor.VisitorSession.__enter__

    def capture(self):
        sessions.append(self)
        return original(self)

    monkeypatch.setattr(visitor.VisitorSession, '__enter__', capture)
    assert app.main(['--pdf-dir', str(tmp_path / 'pdfs'), '--state-dir', str(tmp_path / 'state'),
                     '--embedding-backend', 'tfidf']) == 0
    assert sessions[0].memory._volatile is None
    assert not (tmp_path / 'state' / 'memory.sqlite3').exists()
    if ending in {'reset', 'done', 'finish', 'timeout'}:
        assert seen[-1] == []
    if 'timeout' in ending:
        assert 'inactivity' in capsys.readouterr().out
    if ending == 'idle_without_visitor':
        assert 'inactivity' not in capsys.readouterr().out


def test_real_terminal_idle_timeout(tmp_path):
    master, slave = pty.openpty()
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, str(app.__file__), '--pdf-dir', str(tmp_path / 'pdfs'),
             '--state-dir', str(tmp_path / 'state'), '--embedding-backend', 'tfidf',
             '--idle-timeout', '1'],
            stdin=slave, stdout=slave, stderr=slave)
        # Unsupported university question uses the deterministic fallback, so
        # this exercises the real CLI without a server, weights or network.
        os.write(master, b'What are the tuition fees?\n')
        output = b''
        deadline = time.monotonic() + 10
        while b'Ready for a new visitor.' not in output and time.monotonic() < deadline:
            if select.select([master], [], [], 0.2)[0]:
                output += os.read(master, 4096)
        assert b'Conversation cleared after inactivity.' in output, output
        assert process.poll() is None
        os.write(master, b'/exit\n')
        process.wait(timeout=5)
        assert process.returncode == 0
        assert not (tmp_path / 'state' / 'memory.sqlite3').exists()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(slave)
        os.close(master)


@pytest.mark.parametrize('question,language', [('What are the fees in 2030?', 'en'),
                                              ('Berapa biaya tahun 2030?', 'id')])
def test_unknown_answers_are_brief_and_do_not_narrate_documents(question, language):
    chat = Mock()
    assistant = Assistant(Settings(), Retriever([]), chat=chat)
    answer = assistant.ask(question)
    assert answer.text == FALLBACK[language]
    assert len(answer.text.split()) < 15
    assert answer.sources == []
    chat.answer.assert_not_called()
    assistant.memory.close()


@pytest.mark.parametrize('language', ['en', 'id'])
def test_generated_refusal_without_apology_is_normalized(language):
    text = FALLBACK[language].split(', ', 1)[-1] + ' [1]'
    chat = OllamaChat(Settings(), RecordingClient([{'role': 'assistant', 'content': text}]))
    assert chat.answer('Unsupported question', 'Unrelated facts', [], language) == FALLBACK[language]
