"""Local paths and small-model defaults. No credentials are required."""
from dataclasses import dataclass, field
import os
from pathlib import Path

from embeddings import default_threshold

ROOT = Path(__file__).resolve().parent
BASE_MODEL = 'qwen3:4b-instruct-2507-q4_K_M'
CHAT_MODEL = BASE_MODEL


@dataclass
class Settings:
    model: str = field(default_factory=lambda: os.getenv('MIRAI_MODEL', CHAT_MODEL))
    ollama_host: str = field(default_factory=lambda: os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434'))
    pdf_dir: Path = field(default_factory=lambda: Path(os.getenv('MIRAI_PDF_DIR', str(ROOT / 'data/pdfs'))))
    state_dir: Path = field(default_factory=lambda: Path(os.getenv('MIRAI_STATE_DIR', str(ROOT / '.mirai'))))
    glossary_path: Path = ROOT / 'glossary.json'
    # e5-small: top-3 98% vs 89% for paraphrase-multilingual-MiniLM-L12-v2 on scripts/eval_retrieval.py.
    embedding_model: str = 'intfloat/multilingual-e5-small'
    embedding_backend: str = 'dense'
    embedding_device: str = 'cpu'
    top_k: int = 4
    score_threshold: float | None = None  # None: the embedding model's calibrated default.
    history_turns: int = 6
    history_chars: int = 3000
    context_chars: int = 4500
    num_ctx: int = 4096
    max_tokens: int = 192
    think: bool = False
    keep_alive: str | int = '10m'  # The kiosk server uses -1 so the first visitor never waits for a reload.
    session: str = 'default'
    # small: 20% word errors vs base's 38% on the kiosk recordings, ~4.8 s per clip with beam 1 on 6 cores.
    whisper_model: str = 'small'
    speech_device: str = 'cpu'
    whisper_beam: int = 1  # 5 is about twice as slow; compare with scripts/eval_stt.py --beam.
    whisper_threads: int = 6
    # Below this Whisper score the text has gone far off and the visitor is asked to repeat. Lenient on
    # purpose: small-word slips still read fine to the LLM. Tune with the kiosk's --verbose log.
    min_confidence: float = -1.5
    voices_dir: Path = ROOT / 'models/voices'

    def __post_init__(self):
        if self.score_threshold is None:
            self.score_threshold = default_threshold(self.embedding_model, self.embedding_backend)
