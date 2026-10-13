"""Use the same university vocabulary for ASR, search and pronunciation."""
import json
from pathlib import Path
import re

from config import ROOT


class Glossary:
    def __init__(self, path: Path = ROOT / 'glossary.json'):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        self.terms = data['terms']
        self.corrections = data.get('speech_corrections', {})
        aliases = sorted(
            ((alias, term) for term in self.terms for alias in term['aliases']),
            key=lambda item: len(item[0]), reverse=True,
        )
        self.aliases = {alias.casefold(): term for alias, term in aliases}
        self.pattern = re.compile(
            r'(?<!\w)(?:' + '|'.join(re.escape(alias) for alias, _ in aliases) + r')(?!\w)',
            re.IGNORECASE,
        ) if aliases else None

    def normalize(self, text: str) -> str:
        if self.pattern is None:
            return text
        return self.pattern.sub(lambda match: self.aliases[match.group().casefold()]['canonical'], text)

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
            r'\b(jurusan|major|majors)\b': 'jurusan program studi degree engineering',
            # The FAQ is Indonesian; give English questions the FAQ's own words to match.
            r'\b(scholarships?|grants?)\b': 'beasiswa',
            r'\b(tuition|fees?|costs?|how much|price)\b': 'biaya perkuliahan pembayaran',
            r'\b(installments?|instalments?|credit card|bank loan)\b': 'cicilan bank kartu kredit',
            r'\b(register|registration|apply|application|enrol+|admissions?)\b': 'cara mendaftar pendaftaran',
            r'\b(entrance test|admission test|exam to enter)\b': 'TPKS Tes Potensi Keberhasilan Studi',
            r'\b(clubs?|extracurricular|organi[sz]ations?)\b': 'UKM Unit Kegiatan Mahasiswa',
            r'\b(located|location|address|where is (the )?campus)\b': 'lokasi kampus',
            r'\b(lecturers?|teachers?|professors?|sensei)\b': 'dosen',
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
    def hotwords(self) -> str:
        return ', '.join([term['canonical'] for term in self.terms] + ['Unit Kegiatan Mahasiswa', 'mahasiswa', 'kampus'])

    @property
    def initial_prompt(self) -> str:
        return 'University questions in Indonesian or English. Vocabulary: ' + self.hotwords + '.'

    def pronounce(self, text: str, language: str) -> str:
        text = self.normalize(text)
        for term in self.terms:
            pronunciation = term.get('pronunciation', {}).get(language)
            if pronunciation:
                text = re.sub(r'(?<!\w)' + re.escape(term['canonical']) + r'(?!\w)', pronunciation, text, flags=re.I)
        return text
