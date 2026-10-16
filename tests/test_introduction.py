from unittest.mock import Mock

import pytest

from assistant import Assistant, INTRODUCTION
from config import Settings


@pytest.mark.parametrize('question,language', [
    ('Who are you?', 'en'), ('Hello, who are you?', 'en'),
    ("What's your name?", 'en'), ('Introduce yourself.', 'en'),
    ('What can you do?', 'en'), ('Siapa kamu?', 'id'),
    ('Halo, siapa Anda?', 'id'), ('Kamu siapa?', 'id'),
    ('Apa nama kamu?', 'id'), ('Siapa namamu?', 'id'),
    ('Perkenalkan dirimu.', 'id'), ('Apa tugasmu?', 'id'),
    ('What is MIRAI?', 'en'), ('Apa itu MIRAI?', 'id'), ('MIRAI itu apa?', 'id'), ('Siapa MIRAI?', 'id'),
])
def test_identity_answers_without_pdf_or_generation(question, language):
    retriever, chat = Mock(), Mock()
    assistant = Assistant(Settings(), retriever, chat=chat)
    answer = assistant.ask(question)
    assert answer.text == INTRODUCTION[language]
    assert answer.language == language
    assert answer.sources == []
    retriever.retrieve.assert_not_called()
    chat.answer.assert_not_called()
    assert assistant.memory.recent()[-2]['content'] == question
    assistant.memory.close()


def test_misheard_mirai_is_corrected():
    from glossary import Glossary
    assert Glossary().normalize_speech('Apa itu Miray?') == 'Apa itu MIRAI?'
    assert Glossary().normalize_speech('Halo mi-rai') == 'Halo MIRAI'


def test_identity_obeys_explicit_language_and_preserves_user_name():
    assistant = Assistant(Settings(), Mock(), chat=Mock())
    assistant.memory.append('My name is Raka', 'Hello Raka')
    assert assistant.ask('Who are you?', 'id').text == INTRODUCTION['id']
    assert 'Raka' in assistant.memory.latest_name_turn()[0]['content']
    assistant.memory.clear()
    assert assistant.ask('Siapa kamu?').text == INTRODUCTION['id']
    assistant.memory.close()


def test_combined_university_question_is_not_swallowed_by_introduction():
    retriever, chat = Mock(), Mock()
    retriever.retrieve.return_value = []
    assistant = Assistant(Settings(), retriever, chat=chat)
    answer = assistant.ask('Who are you and what are BINUS ASO tuition fees?')
    assert answer.text != INTRODUCTION['en']
    retriever.retrieve.assert_called_once()
    assistant.memory.close()
