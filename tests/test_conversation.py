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


@pytest.mark.parametrize('reply, spoken', [
    ('Kamu bisa ikut UKM D\'BASE yang membuat mobil hemat energi [1]. Apakah kamu mau tahu lebih lanjut?',
     'Kamu bisa ikut UKM D\'BASE yang membuat mobil hemat energi [1].'),
    ('BASE CORE explores robotics [1]. Want to know more? [1]', 'BASE CORE explores robotics [1].'),
    ('Which program do you mean?', 'Which program do you mean?'),  # a clarification stays
    ('There are four programs [1].', 'There are four programs [1].'),
])
def test_answers_never_end_by_baiting_a_yes(reply, spoken):
    from assistant import strip_trailing_question
    assert strip_trailing_question(reply) == spoken


def assistant_with(hits=()):
    retriever = Mock()
    retriever.retrieve.return_value = list(hits)
    chat = Mock()
    chat.answer.return_value = 'Baik, terima kasih!'
    return Assistant(Settings(), retriever, chat=chat), retriever, chat


@pytest.mark.parametrize('question', ['How are you today?', 'Apa kabar?', 'Good morning!', 'Selamat pagi',
                                      'Are you a robot?', 'What is your favourite food?', 'Tell me a joke',
                                      'Thanks MIRAI!', 'Terima kasih ya', 'Bye!', 'Aku deg-degan', "I'm nervous",
                                      'Aku ada beberapa pertanyaan', 'aku ada beberapa pertanayaan.', 'Boleh tanya?',
                                      'Saya mau tanya', 'I have a few questions.', 'Can I ask you some questions?'])
def test_small_talk_reaches_the_model_without_document_search(question):
    assistant, retriever, chat = assistant_with()
    answer = assistant.ask(question)
    assert answer.text == 'Baik, terima kasih!' and answer.sources == []
    assert chat.answer.call_args.kwargs == {'social': True}
    retriever.retrieve.assert_not_called()


@pytest.mark.parametrize('question', ['Hi, how much does it cost?', 'Halo, ada beasiswa?', 'Thanks! What programs are there?',
                                      'Saya mau tanya, jurusannya apa saja?', 'I have a question about the fees'])
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


def test_evidence_uses_whole_entries_best_first_and_never_a_half_list():
    from assistant import select_context
    big = {'source': 'faq.pdf', 'page': 3, 'text': 'Programs: A, B, C. ' * 30}
    small = {'source': 'faq.pdf', 'page': 4, 'text': 'Clubs: BASE CORE.'}
    text, used = select_context([big, small], max_chars=200)
    assert used == [big] and text.endswith('...'), 'only the first entry may be cut, visibly'
    text, used = select_context([small, big], max_chars=200)
    assert used == [small] and 'Programs' not in text, 'a later entry that does not fit is skipped'


def test_advice_pins_the_program_overview_and_a_related_club():
    from assistant import PROGRAM_OVERVIEW_QUERY
    hits = {'registration': {'source': 'faq.pdf', 'page': 6, 'chunk_id': 1, 'text': 'Daftar.', 'score': 0.81},
            'majors': {'source': 'faq.pdf', 'page': 3, 'chunk_id': 2, 'text': 'Jurusan.', 'score': 0.80},
            'clubs': {'source': 'faq.pdf', 'page': 4, 'chunk_id': 3, 'text': 'UKM.', 'score': 0.79}}
    retriever = Mock()
    retriever.retrieve.side_effect = lambda query, top_k: (
        [hits['majors']] if PROGRAM_OVERVIEW_QUERY in query else
        [hits['clubs']] if 'UKM' in query else [hits['registration'], hits['majors']])
    chat = Mock()
    chat.answer.return_value = 'Automotive & Robotics [1], BASE CORE [2].'
    settings = Settings(context_chars=60)  # Room for two short entries only.
    answer = Assistant(settings, retriever, chat=chat).ask('I like robotics, which faculty should I apply to?')
    assert [hit['text'] for hit in answer.sources] == ['Jurusan.', 'UKM.'], 'pinned entries beat a near-tie'


@pytest.mark.parametrize('question,registration', [('How do I apply?', True), ('Which faculty should I apply to?', False),
                                                   ('What is the application process?', True)])
def test_apply_means_registration_only_for_the_process(question, registration):
    from glossary import Glossary
    expanded = Glossary().expand_query(question)
    assert ('cara mendaftar' in expanded) is registration
    if 'faculty' in question:
        assert 'jurusan program studi' in expanded


def test_vague_continuation_is_a_followup_but_a_new_subject_is_not():
    from assistant import conversation_query
    history = [{'role': 'user', 'content': 'Which BINUS ASO program fits robotics?'},
               {'role': 'assistant', 'content': 'Automotive & Robotics Engineering [1].'}]
    assert conversation_query('What about more for learning?', history)[0] != 'What about more for learning?'
    assert conversation_query('What about the weather?', history) == ('What about the weather?', False, False)


@pytest.mark.parametrize('reply,expected', [
    ('BASE CORE is a robotics club [1]. Specific projects are not detailed in the evidence. [1]',
     "BASE CORE is a robotics club [1]. I don't have more details on that yet."),
    ('BASE CORE explores robotics [1]. The provided documents do not list projects. Want to know more?',
     "BASE CORE explores robotics [1]. I don't have more details on that yet. Want to know more?"),
    ('UKM untuk Business Engineering tidak disebutkan dalam dokumen [1].', None),
    ('You must submit your competition documents and take the TPKS [1].', 'unchanged'),
])
def test_replies_never_narrate_evidence_or_documents(reply, expected):
    language = 'id' if 'UKM untuk' in reply else 'en'
    client = RecordingClient([{'role': 'assistant', 'content': reply}])
    result = OllamaChat(Settings(), client).answer('q', '[1] x', [], language)
    assert result == (reply if expected == 'unchanged' else expected or FALLBACK[language])


def test_only_cited_entries_are_shown_as_sources():
    assistant, _, chat = assistant_with([{'source': 'faq.pdf', 'page': 1, 'chunk_id': 1, 'text': 'TPKS.', 'score': 0.8},
                                         {'source': 'faq.pdf', 'page': 2, 'chunk_id': 2, 'text': 'Kuliah.', 'score': 0.7}])
    chat.answer.return_value = 'Yes, I remember what we talked about in this visit.'
    assert assistant.ask('you have memory capabilities in you?').sources == []
    chat.answer.return_value = 'Kuliahnya pakai bahasa Inggris [2].'
    assert [hit['text'] for hit in assistant.ask('Kuliah pakai bahasa apa?').sources] == ['Kuliah.']


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
