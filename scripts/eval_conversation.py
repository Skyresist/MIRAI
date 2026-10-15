"""Play realistic multi-turn visits against the live model and check each reply.

    python scripts/eval_conversation.py                       # default Qwen model
    python scripts/eval_conversation.py --model gemma3:4b     # compare another Ollama model
    python scripts/eval_conversation.py --low-memory -v       # Jetson profile, print every reply

Checks are deliberately simple and objective: grounded facts mention the FAQ's names, unsupported
questions get the apology, small talk is warm (no apology, no citations), the reply language
matches, and spoken length stays short. Read the printed transcript too; tone needs a human eye.
"""
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as cli
from assistant import Assistant
from glossary import Glossary
from llm import FALLBACK, UNIVERSITY_FALLBACK
from memory import Memory

REFUSALS = {*FALLBACK.values(), *UNIVERSITY_FALLBACK.values()}
ENGLISH_WORDS = {'the', 'and', 'is', 'are', 'you', 'to', 'of', 'in', 'for', 'with', 'can', 'there', 'it', 'a', 'your'}
INDONESIAN_WORDS = {'yang', 'dan', 'di', 'ada', 'untuk', 'dengan', 'kamu', 'ini', 'bisa', 'dari', 'itu', 'akan', 'juga'}
# kind: fact (needs any/all keywords), refuse (must apologise), social (warm, fact-free), memory.
VISITS = {
    'id-major-advice': ('id', [
        ('Halo!', 'social', {}),
        ('Ada jurusan apa saja?', 'fact', {'all': ['Automotive', 'Product Design', 'Business']}),
        ('Aku suka mobil dan robot, cocoknya jurusan yang mana?', 'fact', {'all': ['Automotive']}),
        ('Ada UKM yang berhubungan dengan itu?', 'fact', {'any': ["D'BASE", 'BASE CORE', 'AeroBASE'],
                                                          'none': ['di bawah program', 'bagian dari program']}),
        ('Makasih ya!', 'social', {}),
    ]),
    'en-scholarships': ('en', [
        ('Good morning!', 'social', {}),
        ('Are there scholarships?', 'fact', {'all': ['Widia'], 'any': ['ASO Japan Grant', 'AJGS']}),
        ('What are the requirements for the Widia one?', 'fact', {'any': ['TPKS', 'Tes Potensi', 'test', 'achievement', 'prestasi', 'interview']}),
        ('Do I need to speak Japanese before enrolling?', 'fact', {'any': ['no', 'not required', "don't", 'do not', 'not need']}),
        ('Bye!', 'social', {}),
    ]),
    'id-fees-worried': ('id', [
        ('Aku bingung pilih jurusan', 'social-or-fact', {}),
        ('Biaya kuliahnya berapa juta per semester?', 'no-numbers', {}),
        ('Bisa dicicil?', 'fact', {'any': ['BCA', 'BNI', 'CIMB', 'BRI', '0%']}),
        ('Kampusnya di mana?', 'fact', {'all': ['Alam Sutera']}),
    ]),
    'en-offtopic-and-facts': ('en', [
        ('Are you a robot?', 'social', {}),
        ('What is quantum physics?', 'refuse', {}),
        ('What facilities are there?', 'fact', {'any': ['3D Print', 'Library', 'Monozukuri', 'Computer Lab', 'Physics Lab']}),
        ('Who teaches Product Design Engineering?', 'fact', {'any': ['Tak Sensei', 'Tak'], 'none': ['Saito']}),
        ('Is the class size big?', 'fact', {'any': ['28', '30', 'small']}),
    ]),
    # Reported on the Jetson: 'faculty'/'apply' pulled registration and a club was named as a faculty.
    'advice-from-jetson-test': ('en', [
        ('I like robotics, which faculty should I apply to?', 'fact',
         {'all': ['Automotive'], 'none': ['Faculty of Engineering', 'Mechanical Engineering program']}),
        ('ID:Aku suka robot, sebaiknya pilih jurusan apa?', 'fact',
         {'all': ['Automotive'], 'none': ['Teknik Mesin', 'Teknik Elektro']}),
    ]),
    # The visitor taps EN halfway through: earlier Indonesian turns must not pull the reply back.
    'language-switch': ('id', [
        ('Ada beasiswa apa saja?', 'fact', {'any': ['Widia', 'ASO Japan Grant']}),
        ('EN:What student clubs are there?', 'fact', {'any': ["D'BASE", 'Kendo', 'BASE CORE']}),
    ]),
    'en-memory': ('en', [
        ('My name is Raka.', 'memory', {'any': ['Raka']}),
        ('What labs do you have?', 'fact', {'any': ['Lab']}),
        ("What's my name?", 'memory', {'any': ['Raka']}),
    ]),
}


def check(kind, expect, text, language):
    problems = []
    lowered = text.casefold()
    refused = text.strip() in REFUSALS
    words = len(text.split())
    if kind in ('fact', 'memory'):
        if refused:
            problems.append('refused a supported question')
        if expect.get('all') and not all(k.casefold() in lowered for k in expect['all']):
            problems.append(f"missing {[k for k in expect['all'] if k.casefold() not in lowered]}")
        if expect.get('any') and not any(k.casefold() in lowered for k in expect['any']):
            problems.append(f"none of {expect['any']}")
        if expect.get('none') and any(k.casefold() in lowered for k in expect['none']):
            problems.append(f"wrong claim {[k for k in expect['none'] if k.casefold() in lowered]}")
    elif kind == 'refuse' and not refused:
        problems.append('answered an unsupported question')
    elif kind in ('social', 'social-or-fact'):
        if refused:
            problems.append('cold apology to small talk')
        if kind == 'social' and re.search(r'\[\d+\]', text):
            problems.append('citation in small talk')
    elif kind == 'no-numbers' and re.search(r'\b(rp|idr)\b|\d+\s*(juta|ribu|million)', lowered):
        problems.append('invented a fee amount')
    if words > 70:  # Prompts ask for under 40 words (facts) or 60 (advice); lists run a little longer.
        problems.append(f'too long to speak ({words} words)')
    if not refused and len(text.split()) >= 4:
        # Answers quote Indonesian names ("Himpunan Mahasiswa"), so compare common function words.
        # ...and English program names ("School of Engineering", "Automotive & Robotics").
        prose = re.sub(r"[A-Z][\w'-]*(?:\s+(?:of|&|and|dan)?\s*[A-Z][\w'-]*)*", ' ', text)
        tokens = re.findall(r'\w+', prose.casefold())
        english = sum(token in ENGLISH_WORDS for token in tokens)
        indonesian = sum(token in INDONESIAN_WORDS for token in tokens)
        detected = 'id' if indonesian > english else 'en'
        if detected != language:
            problems.append(f'wrong language ({detected})')
    return problems


def main(argv=None):
    parser = cli.parser()
    parser.description = __doc__
    parser.add_argument('-v', dest='show', action='store_true', help='Print every reply, not only failures.')
    parser.add_argument('--only', help='Run one visit by name.')
    arguments = parser.parse_args(argv)
    settings = cli.build_settings(arguments)
    retriever = cli.build_retriever(settings)
    glossary = Glossary(settings.glossary_path)
    passed = total = 0
    latencies = []
    for name, (language, turns) in VISITS.items():
        if arguments.only and name != arguments.only:
            continue
        memory = Memory(session=name)  # Every visit is a fresh visitor.
        assistant = Assistant(settings, retriever, glossary=glossary, memory=memory)
        print(f'\n=== {name} ({language}) ===')
        for question, kind, expect in turns:
            turn_language = language
            if question[:3] in ('EN:', 'ID:'):  # Screen language switched for this turn.
                turn_language, question = question[:2].lower(), question[3:]
            started = time.perf_counter()
            answer = assistant.ask(question, turn_language)
            latencies.append(time.perf_counter() - started)
            problems = check(kind, expect, answer.text, turn_language)
            total += 1
            passed += not problems
            if arguments.show or problems:
                print(f'{"FAIL" if problems else "ok  "} [{latencies[-1]:4.1f}s] Q: {question}\n'
                      f'      A: {answer.text}' + (f'\n      -> {"; ".join(problems)}' if problems else ''))
        memory.close()
    if total:
        latencies.sort()
        print(f'\n{settings.model}: {passed}/{total} turns passed ({passed / total:.0%}); '
              f'median {latencies[len(latencies) // 2]:.1f}s, slowest {latencies[-1]:.1f}s per reply')


if __name__ == '__main__':
    main()
