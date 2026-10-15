"""Hear the Indonesian voice with different settings and pick the most natural one.

    .venv/bin/python scripts/try_voice.py            # plays each preset in turn
    .venv/bin/python scripts/try_voice.py --en       # same for English

Put the winning numbers in VOICE_TUNING in speech.py (or tell the developer which letter won).
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Settings
from glossary import Glossary
from speech import Speech, respell_id

SAMPLES = {
    'id': 'Halo, selamat datang di BINUS ASO! Kami punya beberapa jurusan, misalnya Automotive and '
          'Robotics Engineering. Ada yang ingin kamu tanyakan lagi?',
    'en': 'Hello, welcome to BINUS ASO! We offer several majors, such as Automotive and Robotics '
          'Engineering. Is there anything else you would like to know?',
}
# length_scale: >1 slower. noise_scale: tone variation (Piper default 0.667). noise_w_scale: rhythm
# variation (default 0.8). pause: seconds of silence between sentences.
PRESETS = {
    'A (old: Piper defaults)': {},
    'B (current: slower + pauses)': {'length_scale': 1.1, 'pause': 0.3},
    'C (slower + livelier tone)': {'length_scale': 1.1, 'noise_scale': 0.8, 'noise_w_scale': 0.9, 'pause': 0.3},
    'D (calm, even slower)': {'length_scale': 1.2, 'noise_scale': 0.5, 'pause': 0.4},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--en', action='store_true', help='Try the English voice instead.')
    language = 'en' if parser.parse_args().en else 'id'
    settings = Settings()
    speech = Speech(settings, Glossary())
    for label, tuning in PRESETS.items():
        output = settings.state_dir / 'try-voice.wav'
        speech.synthesize(SAMPLES[language], output, language, tuning)
        print(f'Playing {label}: {tuning}', flush=True)
        speech.play(output)
        input('  Enter = next preset ')
    if language == 'id':
        output = settings.state_dir / 'try-voice.wav'
        print('Loading E (Meta MMS voice, first run downloads ~145 MB) ...', flush=True)
        mms_wav(respell_id(SAMPLES['id']), output)
        print('Playing E (Meta MMS, a different offline model)', flush=True)
        speech.play(output)


def mms_wav(text, output):
    """Meta's MMS Indonesian voice (facebook/mms-tts-ind, CC-BY-NC 4.0): the main offline alternative to Piper."""
    import wave
    import torch
    from transformers import AutoTokenizer, VitsModel
    tokenizer = AutoTokenizer.from_pretrained('facebook/mms-tts-ind')
    model = VitsModel.from_pretrained('facebook/mms-tts-ind')
    with torch.no_grad():
        audio = model(**tokenizer(text, return_tensors='pt')).waveform[0]
    with wave.open(str(output), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(model.config.sampling_rate)
        wav.writeframes((audio.clamp(-1, 1) * 32767).short().numpy().tobytes())


if __name__ == '__main__':
    main()
