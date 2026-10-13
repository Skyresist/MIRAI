"""Conversation history: in memory by default, with optional file-backed storage."""
from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3


class Memory:
    def __init__(self, path: Path | None = None, session: str = 'default'):
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # The kiosk server answers on worker threads; it serialises all access with one lock.
        self._volatile = sqlite3.connect(':memory:', check_same_thread=False) if self.path is None else None
        self.session = session
        with self._connect() as db, db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY, session TEXT NOT NULL,
                question TEXT NOT NULL, answer TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )''')
            db.execute('CREATE INDEX IF NOT EXISTS turns_session ON turns(session, id)')

    @contextmanager
    def _connect(self):
        if self.path is None:
            if self._volatile is None:
                raise RuntimeError('Conversation memory is closed.')
            yield self._volatile
        else:
            with closing(sqlite3.connect(self.path, timeout=10)) as db:
                yield db

    def close(self):
        if self._volatile is not None:
            self._volatile.close()
            self._volatile = None

    def append(self, question: str, answer: str):
        with self._connect() as db, db:
            db.execute('INSERT INTO turns(session, question, answer) VALUES (?, ?, ?)', (self.session, question, answer))

    def recent(self, limit: int = 6, max_chars: int = 6000) -> list[dict]:
        with self._connect() as db:
            rows = db.execute('SELECT question, answer FROM turns WHERE session=? ORDER BY id DESC LIMIT ?',
                              (self.session, max(0, limit))).fetchall()
        messages, size = [], 0
        for question, answer in rows:
            if size + len(question) + len(answer) > max_chars:
                break
            messages[0:0] = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
            size += len(question) + len(answer)
        return messages

    def search(self, query: str, limit: int = 3) -> list[dict]:
        query = query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        with self._connect() as db:
            rows = db.execute('''SELECT question, answer FROM turns
                WHERE session=? AND (question LIKE ? ESCAPE '\\' OR answer LIKE ? ESCAPE '\\')
                ORDER BY id DESC LIMIT ?''', (self.session, f'%{query}%', f'%{query}%', max(1, min(limit, 5)))).fetchall()
        return [{'question': q[:1000], 'answer': a[:2000]} for q, a in rows]

    def latest_name_turn(self, max_chars: int = 900) -> list[dict]:
        """Recover a self-introduction without loading a longer conversation window."""
        patterns = ['%my name is %', "%my name's %", '%nama saya %', '%nama aku %', '%namaku %']
        with self._connect() as db:
            rows = db.execute(
                'SELECT question, answer FROM turns WHERE session=? AND ('
                + ' OR '.join('question LIKE ?' for _ in patterns)
                + ') ORDER BY id DESC LIMIT 1', (self.session, *patterns)).fetchall()
        for question, answer in rows:
            if len(question) + len(answer) <= max_chars:
                return [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
        return []

    def clear(self):
        with self._connect() as db, db:
            db.execute('DELETE FROM turns WHERE session=?', (self.session,))
