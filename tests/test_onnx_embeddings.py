import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import embeddings
from embeddings import OnnxEmbedding
from retriever import Retriever


def encoder_fixture(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'max_seq_length': 128, 'do_lower_case': False}))
    pooling = tmp_path / 'pooling.json'
    pooling.write_text(json.dumps({'pooling_mode_mean_tokens': True, 'word_embedding_dimension': 2}))
    monkeypatch.setattr(embeddings, 'cached_asset', lambda repo, name, revision:
                        str(config if name == 'sentence_bert_config.json' else pooling if name == '1_Pooling/config.json' else tmp_path / name))
    tokenizer = Mock()
    tokenizer.token_to_id.return_value = 1
    tokenizer.encode_batch.side_effect = lambda texts: [SimpleNamespace(
        ids=[11, 12] if text == 'long' else [11, 1],
        attention_mask=[1, 1] if text == 'long' else [1, 0], type_ids=[0, 0]) for text in texts]
    graph = Mock()
    graph.get_inputs.return_value = [SimpleNamespace(name=name) for name in ('input_ids', 'attention_mask', 'token_type_ids')]

    def inference(outputs, inputs):
        assert all(value.dtype == np.int64 for value in inputs.values())
        # A large pad embedding must be excluded by attention_mask during pooling.
        return [np.asarray([[[2., 0.], [0., 2.]] if row[1] else [[2., 0.], [100., 100.]]
                            for row in inputs['attention_mask']], dtype=np.float32)]

    graph.run.side_effect = inference
    monkeypatch.setitem(__import__('sys').modules, 'tokenizers', SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _: tokenizer)))
    monkeypatch.setitem(__import__('sys').modules, 'onnxruntime', SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(), InferenceSession=lambda *args, **kwargs: graph))
    return OnnxEmbedding('test-model', 'pinned-revision'), tokenizer, graph


def test_onnx_pooling_excludes_padding_and_normalizes(tmp_path, monkeypatch):
    encoder, tokenizer, _ = encoder_fixture(tmp_path, monkeypatch)
    result = encoder.encode(['long', ' short '])
    np.testing.assert_allclose(result, [[2 ** -0.5, 2 ** -0.5], [1., 0.]], atol=1e-6)
    assert result.dtype == np.float32
    tokenizer.enable_truncation.assert_called_once_with(max_length=128)
    assert tokenizer.encode_batch.call_args.args[0] == ['long', 'short']


def test_onnx_indexing_batches_are_bounded(tmp_path, monkeypatch):
    encoder, _, graph = encoder_fixture(tmp_path, monkeypatch)
    result = encoder.encode(['long'] * 17)
    assert result.shape == (17, 2)
    assert [call.args[1]['input_ids'].shape[0] for call in graph.run.call_args_list] == [8, 8, 1]


def test_quantized_cache_cannot_reuse_float_model_vectors(tmp_path, monkeypatch):
    model = Mock()
    model.encode.return_value = np.array([[1., 0.]], dtype=np.float32)
    monkeypatch.setattr(Retriever, '_model', lambda self: model)
    documents = [{'source': 'test.pdf', 'page': 1, 'text': 'A document containing enough text to be indexed.'}]
    cache = tmp_path / 'shared-cache.npz'
    Retriever(documents, backend='dense', cache_path=cache)
    quantized = Retriever(documents, backend='onnx', cache_path=cache)
    assert not quantized.cache_hit and model.encode.call_count == 2
    assert Retriever(documents, backend='onnx', cache_path=cache).cache_hit


def test_quantized_encoder_rejects_cuda():
    with pytest.raises(ValueError, match='uses CPU'):
        Retriever([], backend='onnx', device='cuda')
