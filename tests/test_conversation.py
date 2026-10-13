from unittest.mock import Mock

import httpx
import ollama
import pytest

import llm
from assistant import Assistant
from config import Settings
from llm import FALLBACK, UNIVERSITY_FALLBACK, OllamaChat, complete_sentences
from retriever import Retriever
from test_assistant import RecordingClient


def assistant_with(hits=()):
    retriever = Mock()
    retriever.retrieve.return_value = list(hits)
    chat = Mock()
    chat.answer.return_value = 'Baik, terima kasih!'
    return Assistant(Settings(), retriever, chat=chat), retriever, chat


@pytest.mark.parametrize('question', ['How are you today?', 'Apa kabar?', 'Good morning!', 'Selamat pagi',
                                      'Are you a robot?', 'What is your favourite food?', 'Tell me a joke',
                                      'Thanks MIRAI!', 'Terima kasih ya', 'Bye!', 'Aku deg-degan', "I'm nervous"])
def test_small_talk_reaches_the_model_without_document_search(question):
    assistant, retriever, chat = assistant_with()
    answer = assistant.ask(question)
    assert answer.text == 'Baik, terima kasih!' and answer.sources == []
    assert chat.answer.call_args.kwargs == {'social': True}
    retriever.retrieve.assert_not_called()


@pytest.mark.parametrize('question', ['Hi, how much does it cost?', 'Halo, ada beasiswa?', 'Thanks! What programs are there?'])
def test_greeting_plus_university_question_still_uses_evidence(question):
    assistant, retriever, chat = assistant_with([{'source': 'faq.pdf', 'page': 1, 'text': 'Evidence.'}])
    assistant.ask(question)
    retriever.retrieve.assert_called()
    assert 'social' not in chat.answer.call_args.kwargs


def test_feeling_about_university_without_evidence_gets_a_kind_fact_free_reply():
    assistant, _, chat = assistant_with()
    assistant.ask("I'm nervous about university")
    assert chat.answer.call_args.kwargs == {'social': True}


@pytest.mark.parametrize('question,expected', [('Explain quantum physics.', FALLBACK['en']),
                                               ('What are the tuition fees?', UNIVERSITY_FALLBACK['en'])])
def test_factual_questions_without_evidence_never_reach_the_model(question, expected):
    assistant, _, chat = assistant_with()
    assert assistant.ask(question).text == expected
    chat.answer.assert_not_called()


def test_social_prompt_has_no_evidence_and_no_search_tool():
    client = RecordingClient([{'role': 'assistant', 'content': 'Kabar baik!'}])

    def search(query: str) -> str:
        return ''

    def recall(query: str) -> str:
        return ''

    OllamaChat(Settings(), client).answer('Apa kabar?', '', [], 'id',
                                          {'search_documents': search, 'recall_memory': recall}, social=True)
    request = client.requests[0]
    assert 'Do NOT state any fact about BINUS ASO' in request['messages'][0]['content']
    assert 'evidence' not in request['messages'][-1]['content'].lower()
    assert request['tools'] == [recall]


def test_social_factual_slip_is_normalised_to_the_apology():
    client = RecordingClient([{'role': 'assistant', 'content': FALLBACK['en']}])
    assert OllamaChat(Settings(), client).answer('Hi there, explain gravity', '', [], social=True) == FALLBACK['en']


@pytest.mark.parametrize('reply', ['I do not have knowledge of the weather today.',
                                   "Sorry, I don't have information about fixing laptops.",
                                   'Maaf, saya tidak memiliki informasi tentang cuaca.'])
def test_refusals_in_other_words_use_the_standard_apology(reply):
    client = RecordingClient([{'role': 'assistant', 'content': reply}])
    language = 'id' if reply.startswith('Maaf') else 'en'
    assert OllamaChat(Settings(), client).answer('x', 'evidence', [], language) == FALLBACK[language]


def test_cited_answer_that_mentions_missing_detail_is_kept():
    reply = 'I do not have information about 2030 fees, but installments are available via BCA [1].'
    client = RecordingClient([{'role': 'assistant', 'content': reply}])
    assert OllamaChat(Settings(), client).answer('fees?', '[1] x', []) == reply


def test_campus_topics_missing_from_the_faq_point_to_staff():
    assistant, _, chat = assistant_with()
    assert assistant.ask('Is there a dormitory?').text == UNIVERSITY_FALLBACK['en']
    assert assistant.ask('Ada asrama?').text == UNIVERSITY_FALLBACK['id']
    chat.answer.assert_not_called()


class CrashingThenWorkingClient(RecordingClient):
    """Ollama killed by the OOM killer, then restarted by systemd."""
    def __init__(self, failures, error):
        super().__init__([{'role': 'assistant', 'content': 'Halo!'}])
        self.failures, self.error = failures, error

    def chat(self, **kwargs):
        if self.failures:
            self.failures -= 1
            raise self.error
        return super().chat(**kwargs)


@pytest.mark.parametrize('error', [httpx.ConnectError('refused'), ollama.ResponseError('model failed to load', 500)])
def test_answer_survives_an_ollama_restart(monkeypatch, error):
    waits = []
    monkeypatch.setattr(llm.time, 'sleep', waits.append)
    client = CrashingThenWorkingClient(2, error)
    assert OllamaChat(Settings(), client).answer('Halo', '', [], 'id', social=True) == 'Halo!'
    assert waits == list(llm.RETRY_DELAYS[:2])


def test_bad_requests_and_long_outages_still_fail(monkeypatch):
    monkeypatch.setattr(llm.time, 'sleep', lambda seconds: None)
    with pytest.raises(ollama.ResponseError):
        OllamaChat(Settings(), CrashingThenWorkingClient(1, ollama.ResponseError('model not found', 404))).answer('x', '', [])
    with pytest.raises(httpx.ConnectError):
        OllamaChat(Settings(), CrashingThenWorkingClient(9, httpx.ConnectError('down'))).answer('x', '', [])


def test_emoji_are_removed_before_display_and_speech():
    client = RecordingClient([{'role': 'assistant', 'content': 'Halo! Ada yang bisa aku bantu? 😊'}])
    assert OllamaChat(Settings(), client).answer('Halo', '', [], 'id', social=True) == 'Halo! Ada yang bisa aku bantu?'


def test_new_subject_in_a_followup_is_searched_on_its_own():
    assistant, retriever, _ = assistant_with([{'source': 'faq.pdf', 'page': 1, 'text': 'Evidence.'}])
    assistant.memory.append('Ada jurusan apa di BINUS ASO?', 'Automotive & Robotics Engineering [1].')
    assistant.ask('Ada UKM yang berhubungan dengan itu?')
    searched = [call.args[0] for call in retriever.retrieve.call_args_list]
    assert searched[0].startswith('Ada UKM yang berhubungan dengan itu?') and 'Automotive' not in searched[0]
    assert 'Automotive' in searched[1]


def test_reply_cut_by_token_limit_ends_on_a_full_sentence():
    client = RecordingClient([{'role': 'assistant', 'content': 'Ada tiga jurusan [1]. Kamu bisa belajar rob'}])
    client.chat = (lambda original: lambda **kwargs: {**original(**kwargs), 'done_reason': 'length'})(client.chat)
    assert OllamaChat(Settings(), client).answer('Ada jurusan apa?', '[1] x', []) == 'Ada tiga jurusan [1].'
    assert complete_sentences('Short and cut') == 'Short and cut…'


def test_empty_knowledge_still_answers_small_talk(tmp_path):
    chat = Mock()
    chat.answer.return_value = 'Hello!'
    assert Assistant(Settings(state_dir=tmp_path), Retriever([]), chat=chat).ask('Hello!').text == 'Hello!'
