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

from embeddings import model_spec

logger = logging.getLogger(__name__)
DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_REVISION = model_spec(DEFAULT_MODEL)['revision']


QUESTION_LINE = re.compile(r'^(?:Q(?:uestion)?\s*[:\-]|Pertanyaan\s*[:\-]|Apa\b|Bagaimana\b|Di mana\b|Dimana\b|Kapan\b|Siapa\b|Mengapa\b|What\b|How\b|Where\b|When\b|Who\b|Can\b)|\?$', re.I)
# "10 Dimana lokasi BINUS ASO? Saat ini, ..." keeps its answer on the same line.
NUMBERED_QUESTION = re.compile(r'^\d{1,3}\s+(?P<question>[^\n]+\?)\s*(?P<rest>.*)$')
PIECE_SIZE, PIECE_OVERLAP = 420, 100  # ~128 MiniLM tokens: longer pieces are silently truncated.
PARENT_CHARS = 1800


def faq_blocks(documents: list[dict]) -> list[dict]:
    """Group text into whole question/answer blocks, carrying an answer across page breaks."""
    blocks, current = [], None
    previous_source = None
    # In a numbered FAQ, only numbered lines start questions; "Apa ..." inside an answer does not.
    numbered_sources = {document['source'] for document in documents if any(
        NUMBERED_QUESTION.match(line.strip()) for line in document['text'].splitlines())}
    for document in documents:
        if document['source'] != previous_source:
            current, previous_source = None, document['source']
        for line in document['text'].splitlines():
            line = line.strip()
            if not line:
                continue
            numbered = NUMBERED_QUESTION.match(line)
            unnumbered = document['source'] not in numbered_sources and QUESTION_LINE.search(line)
            if numbered or (unnumbered and (current is None or current['answer'])):
                question = numbered['question'] if numbered else line
                current = {'question': question.strip(), 'answer': [], 'source': document['source'],
                           'page': document.get('page')}
                blocks.append(current)
                if numbered and numbered['rest']:
                    current['answer'].append(numbered['rest'])
            elif current is None:
                current = {'question': '', 'answer': [line], 'source': document['source'], 'page': document.get('page')}
                blocks.append(current)
            else:
                current['answer'].append(line)
    for block in blocks:
        block['answer'] = '\n'.join(block['answer'])
    # Table headers such as "NO QUESTIONS ANSWERS" are not knowledge.
    return [block for block in blocks if block['question'] or len(block['answer']) >= 40]


def chunk_documents(documents: list[dict], size: int = PIECE_SIZE, overlap: int = PIECE_OVERLAP) -> list[dict]:
    """Index small question-prefixed pieces; each remembers the whole Q&A it came from."""
    pieces = []
    for parent_id, block in enumerate(faq_blocks(documents)):
        answer, prefix = block['answer'], (block['question'] + '\n' if block['question'] else '')
        starts = range(0, max(len(answer), 1), size - overlap)
        for start in starts:
            text = (prefix + answer[start:start + size]).strip()
            if text:
                pieces.append({'text': text, 'source': block['source'], 'page': block['page'],
                               'chunk_id': len(pieces), 'parent': parent_id, 'offset': start,
                               'question': block['question'], 'answer': answer})
            if start + size >= len(answer):
                break
    return pieces


def parent_text(piece: dict, limit: int = PARENT_CHARS) -> str:
    """Return the full Q&A, or a window around the matching piece for very long answers."""
    answer = piece['answer']
    if len(answer) > limit:
        start = max(0, min(piece['offset'] - 300, len(answer) - limit))
        answer = ('...' if start else '') + answer[start:start + limit] + ('...' if start + limit < len(answer) else '')
    return ((piece['question'] + '\n') if piece['question'] else '') + answer


class Retriever:
    def __init__(self, documents, *, cache_path=None, model_name=DEFAULT_MODEL,
                 backend='dense', device='cpu', threshold=0.25, model_revision=None):
        if backend not in ('dense', 'onnx', 'tfidf'):
            raise ValueError('Embedding backend must be dense, onnx or tfidf')
        if backend == 'onnx' and device != 'cpu':
            raise ValueError('The quantized ONNX embedding backend uses CPU.')
        self.chunks = chunk_documents(documents)
        self.model_name, self.backend, self.device = model_name, backend, device
        self.spec = model_spec(model_name)
        self.model_revision = model_revision or self.spec['revision']
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
            passages = [self.spec['passage'] + text for text in texts]
            self.embeddings = self._model().encode(passages, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
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
            vector = self._model().encode([self.spec['query'] + query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            dense = cosine_similarity(vector, self.embeddings).ravel()
            scores = 0.9 * dense + 0.1 * lexical
        else:
            scores = lexical
        results, parents = [], set()
        for index in np.argsort(-scores, kind='stable'):
            piece = self.chunks[int(index)]
            if scores[index] < self.threshold or len(results) == top_k:
                break
            if piece['parent'] in parents:
                continue  # Several pieces of one answer must not crowd out other answers.
            parents.add(piece['parent'])
            results.append({'text': parent_text(piece), 'source': piece['source'], 'page': piece['page'],
                            'chunk_id': piece['parent'], 'question': piece['question'],
                            'score': float(scores[index])})
        return results


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
