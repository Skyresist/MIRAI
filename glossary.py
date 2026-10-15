"""Use the same university vocabulary for ASR, search and pronunciation."""
import json
from pathlib import Path
import re

from config import ROOT

SEPARATOR = r'[\s,.\-]*'


class Glossary:
    def __init__(self, path: Path = ROOT / 'glossary.json'):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        self.terms = data['terms']
        self.corrections = data.get('speech_corrections', {})
        aliases = sorted(
            ((alias, term) for term in self.terms for alias in term['aliases']),
            key=lambda item: len(item[0]), reverse=True,
        )
        self.aliases = {self._key(alias): term for alias, term in aliases}
        # Whisper writes spelled-out letters many ways ("Ha, em, be." / "ha-em-be" / "haembe"), so the
        # parts of an alias may be joined by any spaces, commas, dots or dashes, or nothing.
        self.pattern = re.compile(
            r'(?<!\w)(?:' + '|'.join(SEPARATOR.join(map(re.escape, alias.split())) for alias, _ in aliases) + r')(?!\w)',
            re.IGNORECASE,
        ) if aliases else None

    @staticmethod
    def _key(text: str) -> str:
        return re.sub(SEPARATOR, '', text.casefold())

    def normalize(self, text: str) -> str:
        if self.pattern is None:
            return text
        return self.pattern.sub(lambda match: self.aliases[self._key(match.group())]['canonical'], text)

    def normalize_speech(self, text: str) -> str:
        for wrong, correct in self.corrections.items():
            text = re.sub(r'(?<!\w)' + re.escape(wrong) + r'(?!\w)', correct, text, flags=re.I)
        return self.normalize(text)

    def expand_query(self, text: str) -> str:
        normalized = self.normalize(text)
        additions = [term['search'] for term in self.terms if re.search(
            r'(?<!\w)' + re.escape(term['canonical']) + r'(?!\w)', normalized, re.I,
        )]
        if re.search(r'unit kegiatan mahasiswa', normalized, re.I):
            additions.append('UKM organisasi mahasiswa student clubs')
        topics = {
            r'\b(fasilitas|facilities)\b': 'fasilitas facilities laboratorium lab library canteen perpustakaan',
            r'\b(akreditasi|accreditation)\b': 'akreditasi accreditation BAN-PT Unggul',
            # Visitors say "faculty"/"fakultas"; the FAQ only says "jurusan"/"program studi".
            r'\b(jurusan|majors?|faculty|faculties|fakultas|prodi)\b': 'jurusan program studi degree engineering',
            # The FAQ is Indonesian; give English questions the FAQ's own words to match.
            r'\b(scholarships?|grants?)\b': 'beasiswa',
            r'\b(tuition|fees?|costs?|how much|price)\b': 'biaya perkuliahan pembayaran',
            r'\b(installments?|instalments?|credit card|bank loan)\b': 'cicilan bank kartu kredit',
            # "How do I apply" is registration; "which faculty should I apply to" is choosing a major.
            r'\b(register|registration|enrol+|admissions?|how (?:do|can|to) (?:i )?apply|apply (?:for|now|online)|'
            r'application (?:form|process))\b': 'cara mendaftar pendaftaran',
            r'\b(entrance test|admission test|exam to enter)\b': 'TPKS Tes Potensi Keberhasilan Studi',
            r'\b(clubs?|extracurricular|organi[sz]ations?)\b': 'UKM Unit Kegiatan Mahasiswa',
            r'\b(located|location|address|where is (the )?campus)\b': 'lokasi kampus',
            r'\b(lecturers?|teachers?|teach(?:es|ing)?|professors?|sensei|who teaches)\b': 'dosen mengajar',
            r'\b(graduates?|alumni|careers?|jobs?)\b': 'alumni lulusan bekerja perusahaan',
            r'\b(midterms?|exams?)\b': 'Ujian Tengah Semester UTS Enrichment Week',
            r'\b(language of instruction|taught in|english classes)\b': 'bahasa pengantar kuliah bahasa Inggris',
            r'\b(class size|classes|atmosphere)\b': 'suasana kampus kelas mahasiswa',
            r'\b(social science|ips|non-science)\b': 'jurusan IPS Sosial Humaniora',
            r'\b(portfolio)\b': 'portofolio proyek',
        }
        additions.extend(value for pattern, value in topics.items() if re.search(pattern, normalized, re.I))
        return ' '.join([normalized, *additions])

    @property
    def initial_prompt(self) -> str:
        # No comma-separated vocabulary list: in the silence after a question Whisper repeated it
        # ("AeroBASE, BASE"). One plain sentence still shows it the letter abbreviations it cannot spell
        # from Indonesian letter names ("ha em be"); other names are fixed afterwards by the aliases.
        return 'Saya mau tanya tentang UKM dan HMB dan TPKS di BINUS ASO.'

    def pronounce(self, text: str, language: str) -> str:
        text = self.normalize(text)
        for term in self.terms:
            pronunciation = term.get('pronunciation', {}).get(language)
            if pronunciation:
                text = re.sub(r'(?<!\w)' + re.escape(term['canonical']) + r'(?!\w)', pronunciation, text, flags=re.I)
        return text
