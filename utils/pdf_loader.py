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


# The FAQ PDF predates the campus move to BSD. Corrected here until BINUS ASO staff update the PDF;
# a sentence that is no longer in the PDF is simply not found.
FAQ_CORRECTIONS = {
    'Saat ini, BINUS ASO berlokasi di Alam Sutera, Tangerang. Namun di pertengahan tahun 2026, '
    'BINUS-ASO akan hadir sebagai kampus baru di Digital Hub (D-Hub), BSD.':
    'BINUS ASO berlokasi di Digital Hub (D-Hub), BSD, Tangerang.',
}


def correct_faq(text: str) -> str:
    for old, new in FAQ_CORRECTIONS.items():
        # PDF extraction may break a sentence across lines anywhere.
        text = re.sub(r'\s+'.join(map(re.escape, old.split())), new, text)
    return text


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
                    text = correct_faq(clean_text(page.extract_text() or '')).strip()
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
