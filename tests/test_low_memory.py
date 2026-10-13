import builtins
from unittest.mock import Mock

import pytest
import numpy as np

import app
from assistant import Assistant
from llm import OllamaChat
from memory import Memory
from retriever import Retriever
from test_assistant import write_pdf


def test_low_memory_pdf_ignores_previous_visitors_without_heavy_imports(tmp_path, monkeypatch, capsys):
    pdfs = tmp_path / 'pdfs'
    pdfs.mkdir()
    write_pdf(pdfs / 'faq.pdf', ['Apa saja UKM di BINUS ASO?', "UKM dalam dokumen uji: D'BASE dan AeroBASE."])
    state = tmp_path / 'state'
    Memory(state / 'memory.sqlite3').append('Nama saya Raka', 'Halo Raka')
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'sentence_transformers'}:
            raise AssertionError('Low-memory mode must not import the embedding stack')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    model = Mock()
    model.encode.side_effect = lambda texts, **kwargs: np.tile(np.array([[1., 0.]], dtype=np.float32), (len(texts), 1))
    monkeypatch.setattr(Retriever, '_model', lambda self: model)
    seen = []

    def answer(chat, question, context, history, language, tools):
        seen.append(chat.settings)
        assert "D'BASE" in context and 'AeroBASE' in context
        assert 'Raka' not in tools['recall_memory']('Raka')
        return "D'BASE dan AeroBASE [1]."

    monkeypatch.setattr(OllamaChat, 'answer', answer)
    assert app.main(['--low-memory', '--pdf-dir', str(pdfs), '--state-dir', str(state),
                     '-q', 'Apa saja UKM di BINUS ASO?']) == 0
    output = capsys.readouterr()
    assert "D'BASE dan AeroBASE [1]." in output.out
    assert 'faq.pdf (p. 1)' in output.out
    assert seen[0].num_ctx == 3072 and seen[0].max_tokens == 160
    assert seen[0].embedding_backend == 'onnx'
    assert Memory(state / 'memory.sqlite3').search('Raka')


def test_low_memory_explicit_limits_override_profile(tmp_path, monkeypatch):
    settings = []
    original = Assistant.__init__

    def capture(self, value, retriever, **kwargs):
        settings.append(value)
        original(self, value, retriever, **kwargs)

    monkeypatch.setattr(Assistant, '__init__', capture)
    monkeypatch.setattr(OllamaChat, 'answer', Mock(return_value='Hello.'))
    assert app.main(['--low-memory', '--pdf-dir', str(tmp_path / 'pdfs'),
                     '--state-dir', str(tmp_path / 'state'), '-q', 'Hello',
                     '--num-ctx', '3072', '--max-tokens', '160', '--top-k', '3']) == 0
    assert (settings[0].num_ctx, settings[0].max_tokens, settings[0].top_k) == (3072, 160, 3)


@pytest.mark.parametrize('arguments', [['--embedding-backend', 'dense'], ['--paperrag-index', 'any-index']])
def test_low_memory_rejects_heavy_search_override(arguments):
    with pytest.raises(SystemExit) as error:
        app.main(['--low-memory', *arguments])
    assert error.value.code == 2
