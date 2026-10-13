"""Local PDF retrieval with provenance, score filtering and reusable embeddings."""
import hashlib
import json
import logging
from pathlib import Path
import re
import tempfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'


def chunk_documents(documents: list[dict], size: int = 1400, overlap: int = 200) -> list[dict]:
    """Keep FAQ blocks together, splitting long blocks without losing page labels."""
    chunks = []
    question = re.compile(r'^(?:Q(?:uestion)?\s*[:\-]|Pertanyaan\s*[:\-]|Apa\b|Bagaimana\b|Di mana\b|Dimana\b|Kapan\b|Siapa\b|Mengapa\b|What\b|How\b|Where\b|When\b|Who\b|Can\b)|\?$', re.I)
    for document in documents:
        blocks, current = [], []
        for line in document['text'].splitlines():
            line = line.strip()
            if not line:
                continue
            if question.search(line) and current:
                blocks.append('\n'.join(current))
                current = []
            current.append(line)
        if current:
            blocks.append('\n'.join(current))
        for block in blocks:
            for start in range(0, len(block), size - overlap):
                text = block[start:start + size].strip()
                if text:
                    chunks.append({'text': text, 'source': document['source'], 'page': document.get('page'), 'chunk_id': len(chunks)})
                if start + size >= len(block):
                    break
    return chunks


class Retriever:
    def __init__(self, documents, *, cache_path=None, model_name=DEFAULT_MODEL,
                 backend='dense', device='cpu', threshold=0.25, model_revision=None):
        if backend not in ('dense', 'onnx', 'tfidf'):
            raise ValueError('Embedding backend must be dense, onnx or tfidf')
        if backend == 'onnx' and device != 'cpu':
            raise ValueError('The quantized ONNX embedding backend uses CPU.')
        self.chunks = chunk_documents(documents)
        self.model_name, self.backend, self.device = model_name, backend, device
        self.model_revision = model_revision or (DEFAULT_REVISION if model_name == DEFAULT_MODEL else None)
        self.threshold, self.model, self.embeddings = threshold, None, None
        self.cache_hit = False
        self.cache_path = Path(cache_path) if cache_path else None
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=True)
        if not self.chunks:
            return
        texts = [chunk['text'] for chunk in self.chunks]
        self.lexical = self.vectorizer.fit_transform(texts)
        if backend == 'tfidf':
            return
        fingerprint = {'version': 2, 'model': model_name, 'revision': self.model_revision, 'chunks': self.chunks}
        if backend == 'onnx':
            fingerprint.update(version=3, backend='onnx')
        self.fingerprint = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        if self.cache_path and self.cache_path.exists():
            try:
                with np.load(self.cache_path, allow_pickle=False) as saved:
                    if str(saved['fingerprint'].item()) == self.fingerprint:
                        vectors = saved['embeddings']
                        if vectors.ndim == 2 and vectors.shape[0] == len(self.chunks) and np.isfinite(vectors).all():
                            self.embeddings, self.cache_hit = vectors, True
            except (OSError, ValueError, KeyError, EOFError) as error:
                logger.warning('Rebuilding invalid embedding cache: %s', error)
        if self.embeddings is None:
            self.embeddings = self._model().encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            self._save_cache()

    def _model(self):
        if self.model is None:
            if self.backend == 'onnx':
                from embeddings import OnnxEmbedding
                self.model = OnnxEmbedding(self.model_name, self.model_revision)
                return self.model
            from sentence_transformers import SentenceTransformer
            options = {'device': self.device, 'revision': self.model_revision}
            try:
                self.model = SentenceTransformer(self.model_name, local_files_only=True, **options)
            except OSError:
                self.model = SentenceTransformer(self.model_name, **options)
        return self.model

    def _save_cache(self):
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.cache_path.parent, suffix='.npz', delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.savez_compressed(handle, fingerprint=self.fingerprint, embeddings=self.embeddings)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(self.cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def has_data(self):
        return bool(self.chunks)

    def retrieve(self, query, top_k=4):
        if not self.has_data() or not query.strip() or top_k < 1:
            return []
        lexical = cosine_similarity(self.vectorizer.transform([query]), self.lexical).ravel()
        if self.backend in ('dense', 'onnx'):
            vector = self._model().encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            dense = cosine_similarity(vector, self.embeddings).ravel()
            scores = 0.9 * dense + 0.1 * lexical
        else:
            scores = lexical
        order = np.argsort(-scores, kind='stable')
        return [{**self.chunks[int(index)], 'score': float(scores[index])}
                for index in order[:top_k] if scores[index] >= self.threshold]


class PaperRAGRetriever:
    """Read a trusted local PaperRAG index using its recorded embedding settings."""
    def __init__(self, index_dir, *, threshold=0.25):
        try:
            from paperrag.config import PaperRAGConfig
            from paperrag.retriever import Retriever as PaperRetriever
        except ImportError as error:
            raise RuntimeError('Install PaperRAG in its own environment; see README.') from error
        index_dir = Path(index_dir)
        snapshot = index_dir / 'config_snapshot.json'
        if not snapshot.is_file():
            raise FileNotFoundError('PaperRAG index must include config_snapshot.json so the embedding model matches.')
        config = PaperRAGConfig.load_snapshot(snapshot)
        config.index_dir = str(index_dir)
        config.retriever.score_threshold = threshold
        self.retriever = PaperRetriever(config)

    def has_data(self):
        return bool(self.retriever.store.chunks)

    def retrieve(self, query, top_k=4):
        return [{'text': hit.text, 'source': Path(hit.file_path).name,
                 'page': None, 'section': hit.section_name, 'chunk_id': hit.chunk_id,
                 'score': float(hit.score)} for hit in self.retriever.retrieve(query, top_k=top_k)]
