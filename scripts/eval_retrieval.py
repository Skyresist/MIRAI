"""Measure retrieval on the real FAQ: does the right Q&A come back for visitor-style questions?

Run after changing chunking, glossary, thresholds or the embedding model:
    python scripts/eval_retrieval.py --backend onnx
Each case names the FAQ number (from the PDF's numbering) that must be retrieved.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Settings
from glossary import Glossary
from retriever import Retriever, faq_blocks
from utils.pdf_loader import load_pdfs

CASES = [
    ('Apa itu BINUS ASO?', 1), ('What is BINUS ASO?', 1),
    ('Kenapa harus pilih BINUS ASO dibanding kampus lain?', 2), ('What makes BINUS ASO different?', 2),
    ('Apa kelebihan kuliah di sini?', 3), ('What are the advantages of studying here?', 3),
    ('ASO College itu di mana?', 4), ('Where is ASO College in Japan?', 4),
    ('Ada jurusan apa aja?', 5), ('What majors do you offer?', 5),
    ('What do students learn in Automotive and Robotics Engineering?', 5),
    ('Apakah ada beasiswa?', 6), ('Are there scholarships?', 6),
    ('Syarat Widia Scholarship apa?', 7), ('Who can apply for the Widia scholarship?', 7),
    ('Berapa potongan ASO Japan Grant?', 8), ('What does the AJGS scholarship cover?', 8),
    ('Ada UKM apa saja?', 9), ('What student clubs are there?', 9), ('Is there a kendo club?', 9),
    ("What is D'BASE?", 9), ('Ada klub robotik?', 9),
    ('Kampusnya di mana?', 10), ('Where is the campus located?', 10),
    ('Bisa kerja di Jepang setelah lulus?', 11), ('Can I work in Japan after graduating?', 11),
    ('Alumni kerja di mana?', 12), ('Where do graduates work?', 12),
    ('Suasana kampusnya gimana?', 13), ('How big are the classes?', 13),
    ('Ada dosen dari Jepang?', 14), ('Are there Japanese lecturers?', 14),
    ('Kuliahnya pakai bahasa Inggris?', 15), ('What language are classes taught in?', 15),
    ('Harus bisa bahasa Jepang dulu?', 16), ('Do I need to speak Japanese before enrolling?', 16),
    ('Ada UTS tidak?', 17), ('Is there a midterm exam?', 17),
    ('Fasilitasnya apa saja?', 18), ('Is there a 3D printer lab?', 18), ('Does the campus have a library?', 18),
    ('Bisa bikin portofolio?', 19), ('Can I build a portfolio while studying?', 19),
    ('Anak IPS bisa daftar?', 20), ('Can social science students apply?', 20),
    ('Gimana cara daftar?', 21), ('How do I register?', 21),
    ('Berapa biaya kuliahnya?', 22), ('How much is the tuition fee?', 22),
    ('Akreditasinya apa?', 23), ('Is BINUS ASO accredited?', 23),
    ('TPKS itu apa?', 24), ('What is the entrance test?', 24),
    ('Bisa cicilan?', 25), ('Can I pay in installments?', 25),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['dense', 'onnx', 'tfidf'], default='onnx')
    parser.add_argument('--model', default=Settings().embedding_model)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--threshold', type=float, help="Default: the model's calibrated threshold.")
    parser.add_argument('--verbose', '-v', action='store_true', help='Print every case, not only misses.')
    args = parser.parse_args(argv)
    settings = Settings(embedding_model=args.model, embedding_backend=args.backend, score_threshold=args.threshold)
    documents = load_pdfs(settings.pdf_dir)
    questions = [block['question'] for block in faq_blocks(documents)]
    if len(questions) < 25:
        raise SystemExit(f'Expected the 25-question FAQ in {settings.pdf_dir}, found {len(questions)} blocks.')
    retriever = Retriever(documents, backend=args.backend, threshold=settings.score_threshold, model_name=args.model,
                          cache_path=settings.state_dir / f"eval-{args.backend}-{args.model.rsplit('/', 1)[-1]}.npz")
    glossary = Glossary(settings.glossary_path)
    top1 = topk = 0
    for query, number in CASES:
        hits = retriever.retrieve(glossary.expand_query(query), top_k=args.top_k)
        got = [questions.index(hit['question']) + 1 for hit in hits]
        top1 += bool(got) and got[0] == number
        topk += number in got
        if args.verbose or not got or got[0] != number:
            mark = 'ok ' if got and got[0] == number else ('k  ' if number in got else 'MISS')
            scores = ', '.join(f"{n}:{hit['score']:.2f}" for n, hit in zip(got, hits))
            print(f'{mark} want {number:>2} got [{scores}]  {query}')
    print(f'\n{args.backend} {args.model}: top-1 {top1}/{len(CASES)} ({top1 / len(CASES):.0%}), '
          f'top-{args.top_k} {topk}/{len(CASES)} ({topk / len(CASES):.0%})')


if __name__ == '__main__':
    main()
