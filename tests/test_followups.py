from unittest.mock import Mock

import pytest

from assistant import Assistant, CLARIFY_TOPIC, conversation_query
from config import Settings
from llm import FALLBACK, UNIVERSITY_FALLBACK, OllamaChat
from retriever import Retriever
from test_assistant import RecordingClient


def turns(question, answer):
    return [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]


@pytest.mark.parametrize('question', ['Which one involves cars?', 'What facilities does that program use?',
                                   'How much does it cost?', 'Ada fasilitas apa di sana?',
                                   'Yang mana belajar mobil?', 'What about the fees?'])
def test_followup_keeps_actual_previous_topic_for_retrieval(question):
    history = turns('What study programs are available at BINUS ASO?',
                    'Test-only Automotive Engineering and Robotics Engineering. [1]')
    query, university, needs_topic = conversation_query(question, history)
    assert query.startswith(question)
    assert 'Automotive Engineering' in query
    assert 'BINUS ASO' in query
    assert university and not needs_topic


def test_followup_chain_and_subject_switch():
    history = turns('What programs are available at BINUS ASO?', 'Automotive and Robotics [1].')
    history += turns('Which one involves cars?', 'Automotive Engineering [1].')
    query, university, _ = conversation_query('What facilities does that program use?', history)
    assert 'Automotive Engineering' in query and university
    question = 'What is quantum physics?'
    assert conversation_query(question, history) == (question, False, False)
    assert conversation_query('What about the weather?', history) == ('What about the weather?', False, False)
    assert conversation_query('Apa itu mekanika kuantum?', history) == ('Apa itu mekanika kuantum?', False, False)
    assert conversation_query('Hi there', history) == ('Hi there', False, False)
    history += turns(question, FALLBACK['en'])
    assert conversation_query('How much?', history)[2]


def test_selected_program_query_does_not_reintroduce_other_programs_from_older_answers():
    history = turns('What programs are available at BINUS ASO?',
                    'Automotive Engineering and Robotics Engineering. Robotics uses Control Systems Lab [1].')
    history += turns('Which one involves cars?', 'Automotive Engineering focuses on cars [1].')
    query, _, _ = conversation_query('What facilities does that program use?', history)
    assert 'Automotive Engineering' in query
    assert 'Robotics' not in query and 'Control Systems Lab' not in query


@pytest.mark.parametrize('question,expected', [
    ('What are the BINUS ASO tuition fees in 2030?', UNIVERSITY_FALLBACK['en']),
    ('Berapa biaya kuliah BINUS ASO tahun 2030?', UNIVERSITY_FALLBACK['id']),
    ('Explain quantum physics.', FALLBACK['en']),
    ('Apa itu mekanika kuantum?', FALLBACK['id']),
    ('Which one?', CLARIFY_TOPIC['en']),
    ('Yang mana?', CLARIFY_TOPIC['id']),
])
def test_missing_information_routes_to_correct_response(question, expected):
    chat = Mock()
    assistant = Assistant(Settings(), Retriever([]), chat=chat)
    answer = assistant.ask(question)
    assert answer.text == expected
    assert answer.sources == []
    chat.answer.assert_not_called()
    assistant.memory.close()


def test_new_visitor_has_no_previous_topic_and_cannot_inherit_staff_redirect():
    assistant = Assistant(Settings(), Retriever([]), chat=Mock())
    assistant.memory.append('What study programs are available at BINUS ASO?', 'Automotive [1].')
    assert assistant.ask('What about the fees?').text == UNIVERSITY_FALLBACK['en']
    assistant.memory.clear()
    assert assistant.ask('How much?').text == CLARIFY_TOPIC['en']
    assert assistant.ask('What is quantum physics?').text == FALLBACK['en']
    assistant.memory.close()


def test_staff_redirect_also_applies_to_insufficient_retrieved_evidence():
    retriever = Mock()
    retriever.retrieve.return_value = [{'source': 'faq.pdf', 'page': 1, 'text': 'Tuition in 2025 was X.'}]
    chat = Mock()
    chat.answer.return_value = FALLBACK['en']
    assistant = Assistant(Settings(), retriever, chat=chat)
    answer = assistant.ask('What is BINUS ASO tuition in 2030?')
    assert answer.text == UNIVERSITY_FALLBACK['en'] and not answer.sources
    assistant.memory.close()


def test_old_staff_refusal_does_not_leak_into_a_new_general_topic():
    client = RecordingClient([{'role': 'assistant', 'content': UNIVERSITY_FALLBACK['en']}])
    chat = OllamaChat(Settings(), client)
    answer = chat.answer('What is quantum physics?', 'Unrelated university facts', [])
    assert answer == FALLBACK['en']


def test_followup_search_uses_selected_program_but_preserves_real_question():
    retriever = Mock()
    retriever.retrieve.return_value = [{'source': 'test.pdf', 'page': 2, 'text': 'Automotive uses the test lab.'}]
    chat = Mock(return_value=None)
    chat.answer.return_value = 'It uses the test lab [1].'
    assistant = Assistant(Settings(), retriever, chat=chat)
    assistant.memory.append('Which BINUS ASO program involves cars?', 'Automotive Engineering [1].')
    assistant.ask('What facilities does that program use?')
    assert 'Automotive Engineering' in retriever.retrieve.call_args.args[0]
    assert chat.answer.call_args.args[0] == 'What facilities does that program use?'
    assert 'Automotive Engineering' in chat.answer.call_args.kwargs['reference']
    assert assistant.memory.recent()[-2]['content'] == 'What facilities does that program use?'
    assistant.memory.close()


@pytest.mark.parametrize('question', ['What programs are there?', 'Are there scholarships?', 'Is there a library?'])
def test_existential_there_is_not_a_followup_reference(question):
    query, _, needs_topic = conversation_query(question, [])
    assert query == question and not needs_topic


def test_history_enrichment_is_bounded():
    history = turns('BINUS ASO ' + 'long question ' * 500, 'Long answer ' * 500 + ' [1]')
    query, university, needs_topic = conversation_query('What facilities does it have?', history)
    assert university and not needs_topic
    assert len(query) <= 1000


def test_selected_program_reference_is_beside_actual_followup_question():
    client = RecordingClient([{'role': 'assistant', 'content': 'Vehicle Dynamics Lab [1].'}])
    chat = OllamaChat(Settings(), client)
    question = 'What facilities does that program use?'
    chat.answer(question, '[1] Automotive uses Vehicle Dynamics Lab.', [],
                reference='Previous answer: Automotive Engineering.')
    request = client.requests[0]['messages'][-1]['content']
    assert question in request
    assert request.index('Previous answer: Automotive Engineering.') > request.index(question)
    assert 'not evidence' in request
