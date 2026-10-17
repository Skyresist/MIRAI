import pytest

from assistant import ELSEWHERE, LOCATION_QUESTION

LOCATION_QUESTIONS = [
    'Kampusnya di mana?', 'Kampusnya dimana?', 'Kampus BINUS ASO ada di mana?', 'BINUS ASO di mana?',
    'Dimana lokasi BINUS ASO?', 'Di mana kampusnya?', 'Lokasi kampusnya di mana ya?', 'Alamat kampus BINUS ASO apa?',
    'Apakah kampusnya di Alam Sutera?', 'Kampusnya masih di Alam Sutera?', 'Apakah BINUS ASO di BSD?',
    'Where is the campus?', 'Where is the campus located?', "Where's BINUS ASO?", 'Where are you located?',
    'Is the campus in Alam Sutera?', 'What is the campus address?', 'What is the location of the campus?',
]
OTHER_QUESTIONS = [
    'Di mana saya bisa mendaftar?', 'Where can I register?', 'Ada UKM apa saja?', 'Apa itu ASO College dan di mana lokasinya?',
    'Where is ASO College?', 'Kampus di Jepang di mana?', 'Bagaimana cara mendaftar ke BINUS ASO?', 'What majors does the campus offer?',
]


@pytest.mark.parametrize('question', LOCATION_QUESTIONS)
def test_campus_location_questions_get_the_fixed_answer(question):
    assert LOCATION_QUESTION.search(question) and not ELSEWHERE.search(question)


@pytest.mark.parametrize('question', OTHER_QUESTIONS)
def test_other_questions_still_reach_the_faq(question):
    assert not (LOCATION_QUESTION.search(question) and not ELSEWHERE.search(question))
