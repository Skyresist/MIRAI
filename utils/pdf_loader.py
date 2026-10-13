"""Extract page-level text and provenance from a local PDF tree."""
import logging
from pathlib import Path
import re
from pypdf import PdfReader

logger = logging.getLogger(__name__)
PUNCTUATION = str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"',
                             '‐': '-', '‑': '-', '–': '-', ' ': ' '})


def clean_text(text: str) -> str:
    """Make typed questions match PDF text: D’BASE == D'BASE, and unglue 'kelas.Di'."""
    text = text.translate(PUNCTUATION)
    text = re.sub(r'[\U0001F300-\U0001FAFF☀-➿]', '', text)  # 🔗 📍 would be read aloud
    text = re.sub(r'(?<=[a-z)])\.(?=[A-Z0-9])', '. ', text)
    return re.sub(r'[ \t]{2,}', ' ', text)


def load_pdfs(folder_path) -> list[dict]:
    root = Path(folder_path)
    if not root.is_dir():
        logger.warning('PDF directory is missing: %s', root)
        return []
    documents = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() != '.pdf':
            continue
        try:
            reader = PdfReader(path)
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = clean_text(page.extract_text() or '').strip()
                except Exception as error:
                    logger.warning('Cannot extract %s page %s: %s', path.name, page_number, error)
                    continue
                if text:
                    documents.append({'source': path.relative_to(root).as_posix(), 'page': page_number, 'text': text})
                else:
                    logger.warning('No text in %s page %s; a scanned PDF needs OCR (e.g. PaperRAG).', path.name, page_number)
        except Exception as error:
            logger.warning('Cannot read PDF %s: %s', path.name, error)
    logger.info('Loaded %d text pages from %s', len(documents), root)
    return documents
