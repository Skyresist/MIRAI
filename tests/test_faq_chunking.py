from unittest.mock import Mock

import numpy as np

from retriever import Retriever, chunk_documents, faq_blocks, parent_text
from utils.pdf_loader import clean_text, correct_faq

# Same layout as the real BINUS ASO FAQ: numbered questions, a table header and an
# answer that starts on the question's own line (FAQ 10).
PAGES = [
    {'source': 'faq.pdf', 'page': 1, 'text': 'NO QUESTIONS ANSWERS\n9 Apa saja UKM?\n'
     'Beberapa UKM: D’BASE mobil hemat energi.\n' + 'Kendo Club adalah UKM bela diri. ' * 40},
    {'source': 'faq.pdf', 'page': 2, 'text': 'BASE CORE untuk robotika.\n'
     '10 Dimana lokasi BINUS ASO? Saat ini di Alam Sutera, Tangerang.\n'
     '11 Apakah ada beasiswa?\nApa saja beasiswanya? Widia Scholarship.'},
]


def test_numbered_faq_keeps_questions_whole_across_pages():
    blocks = faq_blocks(PAGES)
    assert [block['question'] for block in blocks] == ['Apa saja UKM?', 'Dimana lokasi BINUS ASO?', 'Apakah ada beasiswa?']
    assert 'BASE CORE' in blocks[0]['answer'], 'page-2 continuation belongs to FAQ 9'
    assert blocks[1]['answer'] == 'Saat ini di Alam Sutera, Tangerang.'
    assert 'Apa saja beasiswanya?' in blocks[2]['answer'], 'a question-like answer line is not a new FAQ'


def test_pieces_fit_the_embedding_window_and_carry_their_question():
    pieces = chunk_documents(PAGES)
    assert max(len(piece['text']) for piece in pieces) < 600
    assert all(piece['text'].startswith(piece['question']) for piece in pieces)
    assert len({piece['parent'] for piece in pieces}) == 3


def test_retrieval_returns_each_parent_once_with_the_full_answer(monkeypatch):
    model = Mock()
    # Every piece looks identical to the encoder, so ties must still not repeat one parent.
    model.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2)
    monkeypatch.setattr(Retriever, '_model', lambda self: model)
    hits = Retriever(PAGES, threshold=0.0).retrieve('UKM robotika', top_k=3)
    assert len({hit['chunk_id'] for hit in hits}) == len(hits) == 3
    ukm = next(hit for hit in hits if hit['question'] == 'Apa saja UKM?')
    assert ukm['text'].startswith('Apa saja UKM?') and ukm['page'] == 1


def test_long_answers_are_windowed_around_the_matching_piece():
    piece = {'question': 'Q?', 'answer': 'a' * 3000 + 'TARGET' + 'b' * 3000, 'offset': 3000}
    text = parent_text(piece, limit=1000)
    assert 'TARGET' in text and len(text) < 1100 and text.startswith('Q?\n...')


def test_pdf_text_cleanup_matches_typed_questions():
    assert clean_text('D’BASE kelas.Di sini Eco‑marathon \U0001F517 link') == "D'BASE kelas. Di sini Eco-marathon link"
    assert clean_text('Revolution 4.0 dan aso-group.jp') == 'Revolution 4.0 dan aso-group.jp'


def test_stale_campus_location_is_corrected_even_across_line_breaks():
    page = ('10 Dimana lokasi BINUS ASO? Saat ini, BINUS ASO berlokasi di Alam Sutera,\nTangerang. Namun di '
            'pertengahan tahun 2026, BINUS-ASO akan hadir sebagai kampus baru di Digital Hub (D-Hub), BSD. 11 Apakah')
    assert correct_faq(page) == '10 Dimana lokasi BINUS ASO? BINUS ASO berlokasi di Digital Hub (D-Hub), BSD, Tangerang. 11 Apakah'
