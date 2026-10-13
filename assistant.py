"""University assistant: retrieve facts, use bounded memory and answer via Ollama."""
from dataclasses import dataclass, field
import json
import re

from config import Settings
from glossary import Glossary
from llm import FALLBACK, UNIVERSITY_FALLBACK, OllamaChat, detect_language, is_memory_request
from memory import Memory


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    language: str = 'en'


INTRODUCTION = {
    'en': "I'm MIRAI, an AI assistant for BINUS ASO School of Engineering. "
          'I help answer university questions in English and Indonesian using the available FAQ.',
    'id': 'Saya MIRAI, asisten AI untuk BINUS ASO School of Engineering. '
          'Saya membantu menjawab pertanyaan tentang universitas dalam bahasa Indonesia dan Inggris berdasarkan FAQ yang tersedia.',
}
IDENTITY_QUESTION = re.compile(
    r"(?:(?:hi|hello|halo|hai)[,!]?\s+)?(?:who are you|what are you|"
    r"what(?: is|'s) your name|introduce yourself|what can you do|"
    r"siapa (?:kamu|anda)|(?:kamu|anda) siapa|(?:apa|siapa) nama (?:kamu|anda)|"
    r"nama (?:kamu|anda) siapa|siapa namamu|perkenalkan (?:diri|dirimu)|apa tugasmu)[?!. ]*", re.I)


def build_context(results: list[dict], max_chars: int = 9000) -> str:
    parts, remaining = [], max_chars
    for index, result in enumerate(results, start=1):
        location = f"page {result['page']}" if result.get('page') else result.get('section', 'document')
        header = f"[{index}] {result['source']} | {location}\n"
        if remaining <= len(header) + 40:
            break
        text = result['text'][:remaining - len(header) - 2]
        parts.append(header + text)
        remaining -= len(header) + len(text) + 2
    return '\n\n'.join(parts)


UNIVERSITY_TERMS = re.compile(
    r"\b(binus|aso|university|universities|college|campus|tuition|admissions?|"
    r"scholarships?|accreditation|curriculum|majors?|student clubs?|study programs?|"
    r"universitas|kampus|kuliah|jurusan|mahasiswa|pendaftaran|beasiswa|akreditasi|"
    r"kurikulum|ukm|aerobase|d'base)\b", re.I)
RELATED_TERMS = re.compile(
    r'\b(programs?|courses?|fees?|costs?|facilities|labs?|requirements?|duration|'
    r'program|biaya|fasilitas|laboratorium|syarat|persyaratan|lama)\b', re.I)
REFERENCES = re.compile(
    r'\b(it|its|they|them|their|there|those|which one|which ones|that program|'
    r'that course|that major|that club|that lab|that facility|di sana|disana|'
    r'itu|tersebut|yang mana|yang tadi)\b', re.I)
ELLIPSIS = re.compile(
    r'^(and|what about|how about|bagaimana dengan|kalau|lalu|terus)\b', re.I)
SHORT_FOLLOWUP = re.compile(
    r'^(how much|how long|where|when|berapa|berapa lama|di mana|dimana|kapan)[?!. ]*$', re.I)
# Small talk is recognised positively. Anything not clearly social and without FAQ evidence still
# gets the fixed apology, so a missed pattern can only make MIRAI too polite, never make up a fact.
SOCIAL = re.compile(
    r"^\W*(?:(?:hi|hello|hey|halo|hai|hallo|pagi|siang|sore|malam|selamat \w+|good (?:morning|afternoon|evening)|"
    r"assalamu'?alaikum|permisi)\b[\w\s,!.']*$|"
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
                    or (ELLIPSIS.search(question) and RELATED_TERMS.search(question)))
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
    def __init__(self, settings: Settings, retriever, *, memory=None, chat=None, glossary=None):
        self.settings, self.retriever = settings, retriever
        self.memory = memory if memory is not None else Memory(session=settings.session)
        self.chat = chat or OllamaChat(settings)
        self.glossary = glossary or Glossary(settings.glossary_path)

    def ask(self, question: str, language: str | None = None) -> Answer:
        question = self.glossary.normalize(question.strip())
        if not question:
            raise ValueError('Please enter a question.')
        if len(question) > 4000:
            raise ValueError('Please keep your question under 4000 characters.')
        language = language or ('id' if detect_language(question) == 'Bahasa Indonesia' else 'en')
        if language not in ('id', 'en'):
            raise ValueError('Answer language must be id or en')
        if IDENTITY_QUESTION.fullmatch(question):
            text = INTRODUCTION[language]
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
        social = (bool(SOCIAL.match(question)) and not is_memory_request(question)
                  and not UNIVERSITY_TERMS.search(question) and not RELATED_TERMS.search(question))
        # "How's it going?" contains "it" but is not a follow-up about the previous program.
        query, university, needs_topic = (question, False, False) if social else conversation_query(question, history)
        fallback = UNIVERSITY_FALLBACK[language] if university else FALLBACK[language]
        sources = []

        def search_documents(query: str) -> str:
            """Search local university PDFs. Use for university facts or a follow-up topic."""
            if not isinstance(query, str) or not query.strip() or len(query) > 2000:
                return json.dumps({'error': 'query must contain 1 to 2000 characters'})
            hits = self.retriever.retrieve(self.glossary.expand_query(query), top_k=self.settings.top_k)
            keys = {(hit['source'], hit.get('page'), hit.get('chunk_id')) for hit in sources}
            for hit in hits:
                key = (hit['source'], hit.get('page'), hit.get('chunk_id'))
                if key not in keys and len(sources) < self.settings.top_k * 2:
                    sources.append(hit)
                    keys.add(key)
            return build_context(sources, self.settings.context_chars) or 'No relevant PDF evidence found.'

        def recall_memory(query: str) -> str:
            """Search earlier messages in this session. These are conversation, not verified university facts."""
            if not isinstance(query, str) or not query.strip() or len(query) > 200:
                return json.dumps({'error': 'query must contain 1 to 200 characters'})
            return json.dumps(self.memory.search(query), ensure_ascii=False)

        skip_search = social or needs_topic or is_memory_request(question)
        context = '' if skip_search else search_documents(query[:2000])
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
                text = self.chat.answer(question, context, history, language, tools,
                                        reference=query[len(question):].strip())
            else:
                text = self.chat.answer(question, context, history, language, tools)
        if text.strip() in (FALLBACK[language], UNIVERSITY_FALLBACK[language]):
            text = fallback
        self.memory.append(question, text)
        if text.strip() in (FALLBACK[language], UNIVERSITY_FALLBACK[language], CLARIFY_TOPIC[language]):
            sources = []
        return Answer(text=text, sources=sources, language=language)
