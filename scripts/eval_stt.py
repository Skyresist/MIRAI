"""Compare Whisper sizes on real recordings from the robot's microphone.

1. Record questions on the Jetson, e.g.:  arecord -f S16_LE -r 16000 -c 1 -d 5 recordings/q01.wav
2. Write what was said in recordings/transcripts.tsv:  q01.wav<TAB>id<TAB>Apa saja UKM di BINUS ASO?
3. python scripts/eval_stt.py recordings --models base small

Prints word error rate (lower is better) and seconds per clip for each model. Glossary corrections
are applied first, exactly as the kiosk does, so "binus asso" -> "BINUS ASO" counts as correct.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Settings
from glossary import Glossary
from speech import Speech


def words(text):
    return re.findall(r"[\w']+", text.casefold())


def word_errors(reference, hypothesis):
    """Word-level edit distance (substitutions + insertions + deletions)."""
    ref, hyp = words(reference), words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, hyp_word in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (ref_word != hyp_word))
    return row[-1], len(ref)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--models', nargs='+', default=['base', 'small'])
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = parser.parse_args(argv)
    cases = [line.split('\t') for line in (args.folder / 'transcripts.tsv').read_text(encoding='utf-8').splitlines()
             if line.strip() and not line.startswith('#')]
    glossary = Glossary()
    for model in args.models:
        speech = Speech(replace(Settings(), whisper_model=model, speech_device=args.device), glossary)
        speech.load_whisper()  # Exclude one-time loading from the per-clip timing.
        errors = total = 0
        seconds = []
        print(f'\n=== whisper {model} ===')
        for filename, language, expected in cases:
            started = time.perf_counter()
            try:
                heard, _ = speech.transcribe(args.folder / filename, language)
            except ValueError:
                heard = ''
            seconds.append(time.perf_counter() - started)
            wrong, count = word_errors(glossary.normalize(expected), heard)
            errors, total = errors + wrong, total + count
            print(f'{wrong:2d}/{count:<2d} {seconds[-1]:4.1f}s  {filename}: {heard!r}')
        print(f'whisper {model}: WER {errors / max(total, 1):.1%}, {sum(seconds) / len(seconds):.1f}s per clip')


if __name__ == '__main__':
    main()
