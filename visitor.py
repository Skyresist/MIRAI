"""Visitor lifecycle and idle timeout for the Jetson's Linux terminal."""
import select
import sys

from memory import Memory


class IdleTimeout(Exception):
    pass


def timed_input(prompt, timeout):
    # Canonical terminal input is ready when Enter is pressed. Redirected input
    # and test doubles use normal input; the kiosk timeout applies to a terminal.
    if sys.stdin.isatty():
        print(prompt, end='', flush=True)
        if not select.select([sys.stdin], [], [], timeout)[0]:
            raise IdleTimeout
        return input()
    return input(prompt)


class VisitorSession:
    def __init__(self, timeout=300, label='default'):
        self.timeout = timeout
        self.memory = Memory(session=label)
        self.language = 'en'
        self.prompt = 'You: '
        self.active = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.memory.clear()
        finally:
            self.memory.close()

    def read(self):
        return timed_input(self.prompt, self.timeout).strip()

    def reset(self):
        self.memory.clear()
        self.language = 'en'
        self.prompt = 'You: '
        self.active = False

    def after_answer(self, language):
        self.language = language
        self.active = True
        message = ('Ketik done atau finish jika sudah selesai, atau ajukan pertanyaan lain.'
                   if language == 'id' else
                   'Type done or finish when you are finished, or ask another question.')
        self.prompt = f'Bot: {message}\nYou: '

    def command(self, question):
        command = question.strip().casefold().rstrip('.!?')
        if command == '/exit':
            return 'exit'
        if command in {'done', 'finish', 'finished', 'selesai', 'reset', '/reset',
                       '/done', 'exit', 'bye', 'end chat', 'keluar'}:
            self.reset()
            return 'reset'
        return None
