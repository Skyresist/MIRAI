"""Quantized multilingual semantic embeddings without importing PyTorch."""
import json
from pathlib import Path

import numpy as np

ONNX_MODEL_FILE = 'onnx/model_qint8_arm64.onnx'
ONNX_FILES = [ONNX_MODEL_FILE, 'tokenizer.json', 'sentence_bert_config.json', '1_Pooling/config.json']


def cached_asset(repo, filename, revision):
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo, filename, revision=revision, local_files_only=True)
    except OSError:
        return hf_hub_download(repo, filename, revision=revision)


class OnnxEmbedding:
    def __init__(self, model_name, revision):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        paths = {name: cached_asset(model_name, name, revision) for name in ONNX_FILES}
        config = json.loads(Path(paths['sentence_bert_config.json']).read_text())
        pooling = json.loads(Path(paths['1_Pooling/config.json']).read_text())
        if not pooling.get('pooling_mode_mean_tokens') or any(pooling.get(key) for key in (
                'pooling_mode_cls_token', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens')):
            raise ValueError('This ONNX encoder requires mean-token pooling.')
        self.dimension = pooling['word_embedding_dimension']
        self.lowercase = config.get('do_lower_case', False)
        self.tokenizer = Tokenizer.from_file(paths['tokenizer.json'])
        pad_id = self.tokenizer.token_to_id('<pad>')
        if pad_id is None:
            raise ValueError('Embedding tokenizer has no <pad> token.')
        self.tokenizer.enable_truncation(max_length=config['max_seq_length'])
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token='<pad>')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(paths[ONNX_MODEL_FILE], sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.inputs = {value.name for value in self.session.get_inputs()}
        if (not {'input_ids', 'attention_mask'} <= self.inputs
                or not self.inputs <= {'input_ids', 'attention_mask', 'token_type_ids'}):
            raise ValueError('Unsupported embedding graph inputs.')

    def encode(self, texts, *, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        # Bounded batches keep document indexing from creating large temporary tensors.
        vectors = []
        for start in range(0, len(texts), 8):
            batch = [str(text).strip() for text in texts[start:start + 8]]
            if self.lowercase:
                batch = [text.lower() for text in batch]
            encoded = self.tokenizer.encode_batch(batch)
            values = {
                'input_ids': np.asarray([item.ids for item in encoded], dtype=np.int64),
                'attention_mask': np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                'token_type_ids': np.asarray([item.type_ids for item in encoded], dtype=np.int64),
            }
            hidden = self.session.run(None, {key: values[key] for key in self.inputs})[0]
            if hidden.ndim != 3 or hidden.shape[-1] != self.dimension:
                raise ValueError('Unexpected embedding graph output shape.')
            mask = values['attention_mask'].astype(np.float32)[..., None]
            mean = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
            if normalize_embeddings:
                mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
            vectors.append(mean.astype(np.float32))
        return np.vstack(vectors) if vectors else np.empty((0, self.dimension), dtype=np.float32)
