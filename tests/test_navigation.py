import json
from unittest.mock import Mock

import pytest
import serial

from assistant import Assistant
from config import Settings
from glossary import Glossary
from llm import OllamaChat
from navigation import Command, Navigator, parse
from retriever import Retriever


@pytest.mark.parametrize('text, action, place', [
    ('Bring me to marketing', 'goto', 'marketing'),
    ('take me to the multi-function hall please', 'goto', 'multi_function_hall'),
    ('Bring me to the multi function hall', 'goto', 'multi_function_hall'),
    ('Can you take me to meeting room 3?', 'goto', 'meeting_3'),
    ('bring me to meeting 2', 'goto', 'meeting_2'),
    ('take me to meeting room two', 'goto', 'meeting_2'),
    ('antar aku ke ruang meeting 4', 'goto', 'meeting_4'),
    ('bawa saya ke ruang rapat satu', 'goto', 'meeting_1'),
    ('Get me to cafe', 'goto', 'cafe'),
    ('get me to coffee', 'goto', 'cafe'),
    ('antarkan aku ke kafe dong', 'goto', 'cafe'),
    ('aku mau ke admisi', 'goto', 'admission'),
    ('take me to the admissions office', 'goto', 'admission'),
    ('take me to the meeting room', 'which_meeting', None),
    ('antar aku ke pendaftaran', 'unknown', None),
    ("I'm hungry, take me somewhere to eat", 'unknown', None),
    ('where is the cafe?', 'where', 'cafe'),
    ('di mana ruang marketing?', 'where', 'marketing'),
    ('Patrol', 'patrol', None),
    ('Mirai, start patrolling please', 'patrol', None),
    ('mulai patroli', 'patrol', None),
    ('Back to home', 'home', None),
    ('Mirai go back home', 'home', None),
    ('pulang', 'home', None),
    ('kembali ke tempat awal', 'home', None),
    ('stop', 'stop', None),
    ('Mirai stop!', 'stop', None),
    ('berhenti', 'stop', None),
    ('stop patrol', 'stop', None),
])
def test_commands_are_recognised(text, action, place):
    assert parse(text) == Command(action, place)


@pytest.mark.parametrize('text', [
    'Apa saja UKM di BINUS ASO?',
    'do students go home on weekends?',
    'jam berapa mahasiswa pulang?',
    'apa yang perlu dibawa saat pendaftaran?',
    'show me the scholarship programs',
    'how many students go to BINUS ASO?',
    'Is there a cafe on campus?',
    'Bagaimana cara pendaftaran?',
    'can you stop talking about majors',
    'what is the meeting schedule',
    'take me through the registration steps',
    'where is the campus?',
])
def test_questions_stay_with_the_faq(text):
    assert parse(text) is None


class FakeLink:
    def __init__(self, handler):
        self.handler = handler
        self.seen = []
        self.replies = [b'200 {"stale": true}\n']

    def reset_input_buffer(self):
        self.replies.clear()

    def write(self, data):
        line = data.decode().strip()
        self.seen.append(line)
        code, reply = self.handler(line)
        self.replies += [b'# noise\n', f'{code} {json.dumps(reply)}\n'.encode()]

    def readline(self):
        return self.replies.pop(0) if self.replies else b''

    def close(self):
        pass


def bridge(handler):
    link = FakeLink(handler)
    return link, link.seen


def ok(line):
    return 200, {'ok': True}


def test_trip_is_sent_to_the_bridge():
    link, seen = bridge(ok)
    text, trip = Navigator('/dev/fake', link=link).handle(Command('goto', 'meeting_3'), '', 'en')
    assert seen == ['GOTO meeting_3']
    assert text == 'Sure, follow me to Meeting Room 3!'
    assert trip == {'action': 'goto', 'place': 'meeting_3', 'ok': True}


@pytest.mark.parametrize('action, line', [('home', 'HOME'), ('patrol', 'PATROL'), ('stop', 'STOP')])
def test_home_patrol_and_stop(action, line):
    link, seen = bridge(ok)
    text, trip = Navigator('/dev/fake', link=link).handle(Command(action), '', 'id')
    assert seen == [line] and trip['ok'] and text.startswith('Oke')


def test_place_without_coordinates_is_admitted_politely():
    link, _ = bridge(lambda line: (404, {'error': 'no_spot', 'place': line.split()[1]}))
    text, trip = Navigator('/dev/fake', link=link).handle(Command('goto', 'marketing'), '', 'en')
    assert text == "Sorry, I don't know the way to the Marketing office yet. Please ask the staff nearby."
    assert not trip['ok']


def test_unreachable_robot_says_it_cannot_move():
    navigator = Navigator('/dev/no-such-port')
    text, trip = navigator.handle(Command('home'), '', 'en')
    assert text == "Sorry, I can't move right now. Please ask the staff nearby." and not trip['ok']
    assert navigator.link is None


def test_silent_robot_says_it_cannot_move():
    link = FakeLink(ok)
    link.write = lambda data: None
    text, trip = Navigator('/dev/fake', link=link, timeout=0.2).handle(Command('stop'), '', 'en')
    assert text == "Sorry, I can't move right now. Please ask the staff nearby." and not trip['ok']


def test_broken_cable_reopens_next_time():
    link = FakeLink(ok)

    def unplugged(data):
        raise serial.SerialException('device disconnected')

    link.write = unplugged
    navigator = Navigator('/dev/fake', link=link)
    assert navigator.send('STATUS') == (0, {}) and navigator.link is None


def test_llm_picks_the_place_the_patterns_missed():
    link, seen = bridge(ok)
    chat = Mock()
    chat.choose.return_value = 'cafe'
    text, trip = Navigator('/dev/fake', chat=chat, link=link).handle(
        Command('unknown'), "I'm hungry, take me somewhere to eat", 'en')
    assert seen == ['GOTO cafe'] and trip['place'] == 'cafe'
    assert 'cafe' in chat.choose.call_args.args[2] and 'none' in chat.choose.call_args.args[2]


def test_llm_finding_nothing_moves_nothing():
    link, seen = bridge(ok)
    chat = Mock()
    chat.choose.return_value = 'none'
    text, trip = Navigator('/dev/fake', chat=chat, link=link).handle(
        Command('unknown'), 'take me to the moon', 'en')
    assert seen == [] and text == "Sorry, I don't know the way there. Please ask the staff nearby."


def test_where_and_which_meeting_answer_without_moving():
    link, seen = bridge(ok)
    navigator = Navigator('/dev/fake', link=link)
    text, _ = navigator.handle(Command('where', 'cafe'), '', 'en')
    assert text == 'I can take you to the café. Just say "take me to café" and follow me.'
    text, _ = navigator.handle(Command('which_meeting'), '', 'id')
    assert text == 'Ruang meeting yang mana, 1, 2, 3, atau 4?' and seen == []


def test_choose_keeps_the_model_to_the_list():
    client = Mock()
    client.chat.return_value = {'message': {'content': '{"choice": "cafe"}'}}
    chat = OllamaChat(Settings(), client=client)
    assert chat.choose('system', 'kopi dong', ['cafe', 'none']) == 'cafe'
    assert client.chat.call_args.kwargs['format']['properties']['choice']['enum'] == ['cafe', 'none']
    client.chat.return_value = {'message': {'content': '{"choice": "roof"}'}}
    assert chat.choose('system', 'kopi dong', ['cafe', 'none']) is None
    client.chat.return_value = {'message': {'content': 'not json'}}
    assert chat.choose('system', 'kopi dong', ['cafe', 'none']) is None


def test_assistant_walks_instead_of_searching(tmp_path):
    chat = Mock()
    navigator = Mock()
    navigator.handle.return_value = ('Sure, follow me to the café!', {'action': 'goto', 'place': 'cafe', 'ok': True})
    assistant = Assistant(Settings(state_dir=tmp_path), Retriever([]), chat=chat, glossary=Glossary(),
                          navigator=navigator)
    answer = assistant.ask('Get me to the coffee shop', 'en')
    assert answer.text == 'Sure, follow me to the café!' and answer.navigation['place'] == 'cafe'
    assert navigator.handle.call_args.args[0] == Command('goto', 'cafe')
    chat.answer.assert_not_called()
    assert assistant.memory.recent(2, 1000)[-1]['content'] == 'Sure, follow me to the café!'


def test_empty_nav_port_turns_walking_off(tmp_path):
    chat = Mock()
    chat.answer.return_value = 'Please ask the staff nearby.'
    assistant = Assistant(Settings(state_dir=tmp_path, nav_port=''), Retriever([]), chat=chat, glossary=Glossary())
    assert assistant.navigator is None
    assert assistant.ask('Mirai stop!', 'en').navigation is None
