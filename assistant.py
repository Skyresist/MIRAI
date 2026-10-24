"""University assistant: retrieve facts, use bounded memory and answer via Ollama."""
from dataclasses import dataclass, field
import json
import re

from config import Settings
from glossary import Glossary
from llm import FALLBACK, UNIVERSITY_FALLBACK, OllamaChat, detect_language, is_memory_request
from memory import Memory
import navigation


APOLOGIES = {text for texts in (FALLBACK, UNIVERSITY_FALLBACK) for text in texts.values()}


def without_apologies(history: list[dict]) -> list[dict]:
    """After several apologies in a row the model tacked one onto good answers; they hold no facts."""
    pairs = zip(history[::2], history[1::2])
    return [turn for pair in pairs if pair[1]['content'].strip() not in APOLOGIES for turn in pair]


def strip_trailing_apology(text: str) -> str:
    for apology in APOLOGIES:
        if text.rstrip().endswith(apology) and text.strip() != apology:
            return text.rstrip()[:-len(apology)].rstrip()
    return text


def strip_trailing_question(text: str) -> str:
    """'Mau tahu lebih lanjut?' invites a bare 'iya', which names no topic to search; a 4B model still
    adds one now and then. A reply that is only a question (a clarification) is kept."""
    sentences = re.split(r'(?<=[.!?])\s+(?!\[)', text.strip())
    if len(sentences) > 1 and re.sub(r'(?:\s*\[\d+\])+$', '', sentences[-1]).endswith('?'):
        return ' '.join(sentences[:-1])
    return text


# BINUS ASO has exactly three majors. The model once invented "teknik sipil", so an answer naming a
# major the FAQ never mentions is replaced by the real list. The FAQ does name fields the programs
# combine (Mechanical, Electrical, Computer, Industrial Engineering), so those stay allowed.
PROGRAMS = {
    'en': 'BINUS ASO has three majors: Automotive & Robotics Engineering, Product Design Engineering '
          'and Business Engineering.',
    'id': 'BINUS ASO punya tiga jurusan: Automotive & Robotics Engineering, Product Design Engineering, '
          'dan Business Engineering.',
}
FAQ_ENGINEERING = {'automotive', 'robotics', 'design', 'business', 'mechanical', 'electrical', 'industrial', 'computer', 'of'}
OTHER_MAJOR = re.compile(
    r'\b(?:teknik\s+(?:sipil|kimia|arsitektur|lingkungan|geologi|perkapalan|nuklir|biomedis|pertambangan|'
    r'perminyakan|informatika|telekomunikasi|material|fisika|pangan|pertanian|penerbangan|dirgantara)|'
    r'informatika|ilmu komputer|arsitektur|architecture|kedokteran|medicine|farmasi|pharmacy|'
    r'akuntansi|accounting|psikologi|psychology)\b', re.I)


def names_other_major(text: str) -> bool:
    return bool(OTHER_MAJOR.search(text) or any(
        word.lower() not in FAQ_ENGINEERING for word in re.findall(r'\b([A-Z]\w+)\s+Engineering\b', text)))


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    language: str = 'en'
    navigation: dict | None = None


INTRODUCTION = {
    'en': "I'm MIRAI, an AI guide robot here in the lobby of BINUS ASO School of Engineering. "
          'Just talk to me or type, in English or Indonesian, and ask about majors, scholarships, '
          'student clubs or how to register.',
    'id': 'Saya MIRAI, robot pemandu AI di lobi BINUS ASO School of Engineering. '
          'Kamu bisa bicara atau mengetik dalam bahasa Indonesia atau Inggris, lalu tanya soal jurusan, '
          'beasiswa, UKM, atau cara mendaftar.',
}
# The FAQ still names the old Alam Sutera campus, and the model mixed it up with ASO College in
# Fukuoka, so every question about where this campus is gets one fixed answer.
LOCATION = {
    'en': 'The BINUS ASO campus is at BINUS BSD, Tangerang, Banten.',
    'id': 'Kampus BINUS ASO berada di BINUS BSD, Tangerang, Banten.',
}
LOCATION_QUESTION = re.compile(
    r"\b(?:kampus\w*|campus|binus(?: aso)?)(?: \w+)? (?:ada |berada |terletak |berlokasi )?di ?mana\b|"
    r"\bdi ?mana (?:(?:lokasi|letak|alamat)\w* )?(?:kampus|binus)|"
    r"\b(?:lokasi|letak|alamat)\w* (?:kampus|binus)|"
    r"\bwhere(?: is|'s| are)? (?:the |your |this )?(?:campus|binus|school|university)\b|\bwhere are you (?:located|based)\b|"
    r"\b(?:campus|binus aso)(?:'s)? (?:location|address)\b|\b(?:location|address) of (?:the )?(?:campus|binus)|"
    r"\b(?:kampus\w*|campus|binus(?: aso)?)\b.*\b(?:alam sutera|bsd|d-hub|digital hub|tangerang|jakarta)\b", re.I)
# ASO College's own campus in Japan is a different question; the FAQ answers it.
ELSEWHERE = re.compile(r'\b(?:aso college|jepang|japan|fukuoka|kyushu)\b', re.I)
IDENTITY_QUESTION = re.compile(
    r"(?:(?:hi|hello|halo|hai)[,!]?\s+)?(?:who are you|what are you|"
    r"what(?: is|'s) your name|introduce yourself|what can you do|"
    r"siapa (?:kamu|anda)|(?:kamu|anda) siapa|(?:apa|siapa) nama (?:kamu|anda)|"
    r"nama (?:kamu|anda) siapa|siapa namamu|perkenalkan (?:diri|dirimu)|apa tugasmu|"
    r"(?:what|who) is mirai|tell me about mirai|(?:apa|siapa) (?:itu )?mirai|mirai (?:itu )?(?:apa|siapa))(?: (?:sih|ya|nih))?[?!. ]*", re.I)


def select_context(results: list[dict], max_chars: int = 9000) -> tuple[str, list[dict]]:
    """Whole FAQ entries in the given order; returns the text and the entries actually included.

    An entry that does not fit is skipped, not cut: a list that stops halfway ("Automotive & Robotics,
    ...") makes a small model invent the rest. Only the first entry may be cut, so there is evidence.
    """
    parts, used, remaining = [], [], max_chars
    for result in results:
        location = f"page {result['page']}" if result.get('page') else result.get('section', 'document')
        header = f"[{len(used) + 1}] {result['source']} | {location}\n"
        text = result['text']
        if len(header) + len(text) > remaining:
            if used or remaining <= len(header) + 40:
                continue
            text = text[:remaining - len(header) - 2] + '...'
        parts.append(header + text)
        used.append(result)
        remaining -= len(header) + len(text) + 2
    return '\n\n'.join(parts), used


def source_key(hit: dict) -> tuple:
    return hit['source'], hit.get('page'), hit.get('chunk_id')


def build_context(results: list[dict], max_chars: int = 9000) -> str:
    return select_context(results, max_chars)[0]


UNIVERSITY_TERMS = re.compile(
    r"\b(binus|aso|university|universities|college|campus|tuition|admissions?|"
    r"scholarships?|accreditation|curriculum|majors?|student clubs?|study programs?|"
    r"universitas|kampus|kuliah|jurusan|mahasiswa|pendaftaran|beasiswa|akreditasi|faculty|faculties|fakultas|"
    r"kurikulum|ukm|aerobase|d'base|dorm|dormitory|dormitories|asrama|parking|parkir)\b", re.I)
RELATED_TERMS = re.compile(
    r'\b(programs?|courses?|fees?|costs?|facilities|labs?|requirements?|duration|'
    r'program|biaya|fasilitas|laboratorium|syarat|persyaratan|lama)\b', re.I)
REFERENCES = re.compile(
    r'\b(it|its|they|them|their|there|those|which one|which ones|that program|'
    r'that course|that major|that club|that lab|that facility|di sana|disana|'
    r'itu|tersebut|yang mana|yang tadi)\b', re.I)
ELLIPSIS = re.compile(
    r'^(and|what about|how about|bagaimana dengan|kalau|lalu|terus)\b', re.I)
ADVICE = re.compile(
    r"\b(i like|i love|i enjoy|i'?m interested|interested in|suka|senang|tertarik|hobi|hobby|hobbies|"
    r"cocok|should i|sebaiknya|recommend|rekomendasi|saran|fits? me|suits? me|pilih|memilih|milih|"
    r"pick|choose|decide|tentukan|menentukan)\b", re.I)
INTEREST = re.compile(
    r"\b(i like|i love|i enjoy|i'?m interested|interested in|suka|senang|tertarik|hobi|hobby|hobbies)\b", re.I)
PROGRAM_OVERVIEW_QUERY ='Program studi apa saja yang ditawarkan BINUS ASO? What study programs are offered?'
# Vague continuations that only make sense with the previous topic ("What about more for learning?"),
# unlike a new subject ("What about the weather?").
CONTINUATION = re.compile(
    r'\b(more|else|other|another|learn|learning|study|studying|lagi|lain|lainnya|belajar|selain itu)\b', re.I)
SHORT_FOLLOWUP = re.compile(
    r'^(how much|how long|where|when|berapa|berapa lama|di mana|dimana|kapan)[?!. ]*$', re.I)
# Small talk is recognised positively. Anything not clearly social and without FAQ evidence still
# gets the fixed apology, so a missed pattern can only make MIRAI too polite, never make up a fact.
SOCIAL = re.compile(
    r"^\W*(?:(?:hi|hello|hey|halo|hai|hallo|pagi|siang|sore|malam|selamat \w+|good (?:morning|afternoon|evening)|"
    r"assalamu'?alaikum|permisi)\b[\w\s,!.']*$|"
    # "Aku ada beberapa pertanyaan" announces a question without one; FAQ search on it found unrelated
    # entries and the model invented a program. Whisper may misspell "pertanyaan".
    r"(?:(?:aku|saya)\s+(?:ada|punya|mau)\s+(?:\w+\s+)?(?:perta\w*|tanya\w*)|boleh\s+(?:aku\s+|saya\s+)?tanya\w*|"
    r"mau\s+tanya\w*|i\s+(?:have|got)\s+(?:a\s+|some\s+|a\s+few\s+)?questions?|"
    r"can\s+i\s+ask(?:\s+you)?(?:\s+(?:a|some)\s+questions?)?)[\s,!.?]*$|"
    r".*\b(?:how are you|how(?:'s| is) it going|how do you do|what'?s up|apa kabar|gimana kabar|kabarmu|"
    r"thank(?:s| you)|makasih|terima ?kasih|thx|"
    r"bye|goodbye|see you|sampai jumpa|dadah|daah|"
    r"nice to meet you|senang bertemu|salam kenal|"
    r"are you (?:a robot|human|real|an ai|smart|okay|ok|happy|tired|bored)|kamu (?:robot|manusia|pintar|lucu|keren)|"
    r"(?:do|can) you (?:like|love|eat|sleep|dance|sing|feel|dream)|kamu (?:suka|bisa (?:makan|tidur|nyanyi|menari))|"
    r"your fav(?:ou?rite)?|favoritmu|kesukaanmu|how old are you|umur(?:mu| kamu)|"
    r"tell me a joke|lelucon|joke|cerita lucu|"
    r"you(?:'re| are) (?:cute|cool|smart|great|awesome|funny|helpful)|"
    r"i(?:'m| am) (?:nervous|confused|worried|scared|excited|happy|sad|tired|bored|lost)|"
    r"(?:aku|saya) (?:bingung|takut|gugup|deg-?degan|senang|seneng|sedih|capek|lelah|bosan|khawatir|excited)|"
    r"good job|great job|well done|keren|mantap|lucu banget)\b.*$)", re.I | re.S)
CLARIFY_TOPIC = {
    'en': 'Which topic or program do you mean?',
    'id': 'Topik atau program mana yang Anda maksud?',
}


def conversation_query(question, history):
    """Use bounded, real visitor turns for retrieval; never treat them as evidence."""
    if is_memory_request(question):
        return question, False, False
    university = bool(UNIVERSITY_TERMS.search(question))
    if re.match(r'^(hi|hello|halo|hai)\b', question, re.I) and not university:
        return question, False, False
    # "Apa itu mekanika kuantum?" defines a new topic; its "itu" is not a
    # reference to the previous answer. Bare "Apa itu?" still needs a topic.
    reference_text = re.sub(r'^apa\s+itu\s+(?=\w)', '', question, flags=re.I)
    # "Are there scholarships?" / "What programs are there?": existential "there" points nowhere.
    reference_text = re.sub(r'\b(?:is|are|was|were) there\b|\bthere (?:is|are|was|were)\b', '', reference_text, flags=re.I)
    followup = bool(REFERENCES.search(reference_text) or SHORT_FOLLOWUP.fullmatch(question)
                    or (ELLIPSIS.search(question) and (RELATED_TERMS.search(question) or CONTINUATION.search(question))))
    if not followup:
        return question, university, False
    if len(history) < 2:
        return question, university, not university
    previous = history[-2]['content']
    answer = history[-1]['content']
    _, previous_university, _ = conversation_query(previous, history[:-2])
    previous_university = previous_university or bool(re.search(r'\[\d+\]', answer))
    if not previous_university:
        return question, university, not university
    refusals = {*FALLBACK.values(), *UNIVERSITY_FALLBACK.values(), *CLARIFY_TOPIC.values()}
    answer = '' if answer in refusals else re.sub(r'\[\d+\]', '', answer)
    query = f'{question}\nPrevious answer: {answer[:360]}\nPrevious question: {previous[:220]}'
    return query[:1000], True, False


class Assistant:
    def __init__(self, settings: Settings, retriever, *, memory=None, chat=None, glossary=None, navigator=None):
        self.settings, self.retriever = settings, retriever
        self.memory = memory if memory is not None else Memory(session=settings.session)
        self.chat = chat or OllamaChat(settings)
        self.glossary = glossary or Glossary(settings.glossary_path)
        if navigator is None and settings.nav_url:
            navigator = navigation.Navigator(settings.nav_url, self.chat)
        self.navigator = navigator

    def ask(self, question: str, language: str | None = None) -> Answer:
        question = self.glossary.normalize(question.strip())
        if not question:
            raise ValueError('Please enter a question.')
        if len(question) > 4000:
            raise ValueError('Please keep your question under 4000 characters.')
        language = language or ('id' if detect_language(question) == 'Bahasa Indonesia' else 'en')
        if language not in ('id', 'en'):
            raise ValueError('Answer language must be id or en')
        command = navigation.parse(question) if self.navigator else None
        if command:
            text, trip = self.navigator.handle(command, question, language)
            self.memory.append(question, text)
            return Answer(text=text, language=language, navigation=trip)
        if IDENTITY_QUESTION.fullmatch(question):
            text = INTRODUCTION[language]
            self.memory.append(question, text)
            return Answer(text=text, language=language)
        if LOCATION_QUESTION.search(question) and not ELSEWHERE.search(question):
            text = LOCATION[language]
            self.memory.append(question, text)
            return Answer(text=text, language=language)
        history = self.memory.recent(self.settings.history_turns, self.settings.history_chars)
        if is_memory_request(question) and re.search(r'\b(name|nama|namaku)\b', question, re.I):
            # Keep a name available after older turns leave the short prompt window.
            # These are actual saved messages; no profile fact or assistant turn is invented.
            name_turn = self.memory.latest_name_turn(self.settings.history_chars)
            if name_turn and name_turn[0] not in history:
                reserved = sum(len(turn['content']) for turn in name_turn)
                history = name_turn + self.memory.recent(
                    self.settings.history_turns, self.settings.history_chars - reserved)
        history = without_apologies(history)
        social = (bool(SOCIAL.match(question)) and not is_memory_request(question)
                  and not UNIVERSITY_TERMS.search(question) and not RELATED_TERMS.search(question))
        # "How's it going?" contains "it" but is not a follow-up about the previous program.
        query, university, needs_topic = (question, False, False) if social else conversation_query(question, history)
        fallback = UNIVERSITY_FALLBACK[language] if university else FALLBACK[language]
        sources, cited, pinned = [], [], set()
        # "Help me choose a major" after "aku suka mobil": the 4B model often lost the earlier interest
        # and listed every program, so it travels with the question (search and model; memory keeps the original).
        asked = question
        if ADVICE.search(question) and not INTEREST.search(question):
            interest = next((turn['content'] for turn in reversed(history)
                             if turn['role'] == 'user' and INTEREST.search(turn['content'])), None)
            if interest:
                asked = f'{question} ({interest})'

        def search_documents(query: str) -> str:
            """Search local university PDFs. Use for university facts or a follow-up topic."""
            return search(query)

        def search(query, pin_best=False):
            if not isinstance(query, str) or not query.strip() or len(query) > 2000:
                return json.dumps({'error': 'query must contain 1 to 2000 characters'})
            hits = self.retriever.retrieve(self.glossary.expand_query(query), top_k=self.settings.top_k)
            keys = {source_key(hit) for hit in sources}
            for hit in hits:
                if source_key(hit) not in keys and len(sources) < self.settings.top_k * 3:
                    sources.append(hit)
                    keys.add(source_key(hit))
            if pin_best and hits:
                pinned.add(source_key(hits[0]))
            # Pinned entries, then best match first: the evidence budget goes to what the answer needs
            # rather than to search order or a near-tie.
            sources.sort(key=lambda hit: (source_key(hit) not in pinned, -hit.get('score', 0)))
            text, used = select_context(sources, self.settings.context_chars)
            cited[:] = used  # Citations [1], [2] and the on-screen sources match what the model saw.
            return text or 'No relevant PDF evidence found.'

        def recall_memory(query: str) -> str:
            """Search earlier messages in this session. These are conversation, not verified university facts."""
            if not isinstance(query, str) or not query.strip() or len(query) > 200:
                return json.dumps({'error': 'query must contain 1 to 200 characters'})
            return json.dumps(self.memory.search(query), ensure_ascii=False)

        skip_search = social or needs_topic or is_memory_request(question)
        context = ''
        if not skip_search:
            # Search the visitor's own words first: in "Ada UKM yang berhubungan dengan itu?" the new
            # subject (UKM) must not be drowned out by the previous answer appended for context.
            # In a follow-up the visitor's own words carry the new need ("more for learning?"); pin
            # their best match so the previous answer's FAQs cannot fill the whole evidence budget.
            searches = [(question, True)] if query != question else []
            searches.append((query[:2000], False))
            if ADVICE.search(question):
                # "I like robotics, which major?" needs the overview of all programs (fixed query: the
                # visitor's words pulled in look-alike FAQs) AND the club matching their interest.
                searches += [(PROGRAM_OVERVIEW_QUERY, True),
                             (asked + ' UKM Unit Kegiatan Mahasiswa klub', True)]
            for text, pin_best in searches:
                context = search(text, pin_best)
        memory_or_chat = re.search(
            r'\b(hello|hi|halo|hai|thanks|thank|terima kasih|remember|recall|ingat|tadi|earlier)\b|'
            r'\b(my name|nama saya|nama aku|what did|apa yang (saya|aku)|you said|i am|saya (bernama|suka))\b',
            question, re.I,
        )
        if needs_topic:
            text = CLARIFY_TOPIC[language]
        elif not sources and not (social or SOCIAL.match(question) or memory_or_chat or is_memory_request(question)):
            text = fallback
        else:
            tools = {'search_documents': search_documents, 'recall_memory': recall_memory}
            # "I'm nervous about university" mentions a campus word but is still a feeling;
            # without evidence it gets the fact-free social reply rather than a cold apology.
            if social or (not sources and SOCIAL.match(question) and not is_memory_request(question)):
                text = self.chat.answer(question, context, history, language, tools, social=True)
            elif query != question:
                text = strip_trailing_question(self.chat.answer(question, context, history, language, tools,
                                                                reference=query[len(question):].strip()))
            else:
                text = strip_trailing_question(self.chat.answer(asked, context, history, language, tools))
        text = strip_trailing_apology(text)
        if names_other_major(text):
            text = PROGRAMS[language]
        if text.strip() in (FALLBACK[language], UNIVERSITY_FALLBACK[language]):
            text = fallback
        self.memory.append(question, text)
        if text.strip() in (FALLBACK[language], UNIVERSITY_FALLBACK[language], CLARIFY_TOPIC[language]):
            cited = []
        # Show only entries the reply cites: small talk answered beside unrelated evidence
        # ("you have memory?") must not list "Source: Apa itu TPKS".
        numbers = {int(number) for number in re.findall(r'\[(\d+)\]', text)}
        cited = [hit for index, hit in enumerate(cited, start=1) if index in numbers]
        return Answer(text=text, sources=cited, language=language)
