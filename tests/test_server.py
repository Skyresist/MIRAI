from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from llm import OllamaChat
import server
from test_assistant import write_pdf


@pytest.fixture
def client(tmp_path, monkeypatch):
    pdfs = tmp_path / 'pdfs'
    pdfs.mkdir()
    write_pdf(pdfs / 'faq.pdf', ['1 Apa saja UKM di BINUS ASO?', "UKM dalam dokumen uji: D'BASE dan AeroBASE."])
    seen = []

    def answer(chat, question, context, history, language, tools, **reference):
        seen.append({'question': question, 'history': history, 'keep_alive': chat.settings.keep_alive})
        return "D'BASE dan AeroBASE [1]."

    monkeypatch.setattr(OllamaChat, 'answer', answer)
    speech = SimpleNamespace(
        transcribe=Mock(return_value=('Apa saja UKM?', 'id')),
        synthesize=Mock(side_effect=lambda text, path, language: path.write_bytes(b'RIFF-test-wav')),
    )
    monkeypatch.setattr(server.Kiosk, 'speech', property(lambda self: speech))
    arguments = server.cli.parser().parse_args(['--pdf-dir', str(pdfs), '--state-dir', str(tmp_path / 'state'),
                                                '--embedding-backend', 'tfidf'])
    arguments.no_voice = False
    with TestClient(server.create_app(arguments, preload=False)) as test_client:
        test_client.seen, test_client.speech = seen, speech
        yield test_client


def test_ask_returns_grounded_answer_with_faq_source(client):
    response = client.post('/api/ask', json={'text': 'Apa saja UKM?', 'language': 'id'})
    assert response.status_code == 200
    body = response.json()
    assert body['text'] == "D'BASE dan AeroBASE [1]." and body['language'] == 'id'
    assert body['sources'][0]['question'] == 'Apa saja UKM di BINUS ASO?'
    assert client.seen[0]['keep_alive'] == -1


def test_reset_starts_a_fresh_visitor(client):
    client.post('/api/ask', json={'text': 'Apa saja UKM?', 'language': 'id'})
    client.post('/api/ask', json={'text': 'Apa saja UKM?', 'language': 'id'})
    assert client.seen[-1]['history'], 'second question sees the first'
    assert client.post('/api/reset').json() == {'ok': True}
    client.post('/api/ask', json={'text': 'Apa saja UKM?', 'language': 'id'})
    assert client.seen[-1]['history'] == []


def test_transcribe_passes_screen_language_and_cleans_up(client):
    response = client.post('/api/transcribe', files={'audio': ('speech.webm', b'opus-bytes', 'audio/webm')},
                           data={'language': 'id'})
    assert response.json() == {'text': 'Apa saja UKM?', 'language': 'id'}
    path, language = client.speech.transcribe.call_args.args
    assert language == 'id' and not path.exists()


def test_silence_is_not_a_server_error(client):
    client.speech.transcribe.side_effect = ValueError('No speech detected')
    response = client.post('/api/transcribe', files={'audio': ('speech.webm', b'x', 'audio/webm')})
    assert response.status_code == 200 and response.json()['text'] == ''


def test_speak_returns_wav(client):
    response = client.post('/api/speak', json={'text': 'Halo', 'language': 'id'})
    assert response.status_code == 200 and response.headers['content-type'] == 'audio/wav'


@pytest.mark.parametrize('payload', [{'text': ''}, {'text': 'hi', 'language': 'fr'}])
def test_invalid_questions_are_rejected(client, payload):
    assert client.post('/api/ask', json=payload).status_code == 422


def test_ui_is_served(client):
    page = client.get('/')
    assert page.status_code == 200 and 'MIRAI' in page.text
    assert client.get('/api/status').json()['knowledge'] is True
