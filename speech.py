"""Optional local faster-whisper ASR and Piper VITS/eSpeak speech output."""
from pathlib import Path
import re
import shutil
import subprocess
import wave

from config import Settings
from glossary import Glossary

VOICES = {'id': 'id_ID-news_tts-medium', 'en': 'en_US-lessac-medium'}
# The news voice at Piper defaults reads too fast with no breath between sentences. length_scale > 1
# is slower; noise_* add variation (Piper's own defaults when unset). Compare: scripts/try_voice.py.
VOICE_TUNING = {'id': {'length_scale': 1.1, 'pause': 0.3}}
# The Indonesian voice reads English words with Indonesian letter rules ("Engineering" breaks apart),
# so English terms from the FAQ are respelled the way an Indonesian speaker says them (c = "ch", sy = "sh").
# Add a word here when it sounds wrong; keys are lowercase.
ID_RESPELL = {
    'and': 'en', '&': 'en', 'the': 'de',
    'school': 'skul', 'engineering': 'enjiniring', 'engineer': 'enjinir', 'university': 'yuniversiti',
    'college': 'kolej', 'colleges': 'kolejis', 'group': 'grup',
    'revolution': 'revolusyen', 'hands-on': 'hens on', 'experience': 'ekspiriens', 'project': 'projek',
    'project-based': 'projek beist', 'based': 'beist', 'learning': 'lerning', 'internship': 'internsyip',
    'enrichment': 'enricmen', 'international': 'internesyenal', 'curriculum': 'kurikulum',
    'japanese': 'jepanis', 'language': 'lengwej', 'course': 'kors', 'professional': 'profesyenal',
    'lecturers': 'lekcerers', 'week': 'wik', 'summer': 'samer', 'research': 'riserc', 'study': 'stadi',
    'abroad': 'abrod', 'entrepreneurship': 'ontreprenersyip', 'small': 'smol', 'class': 'kles',
    'method': 'metod', 'station': 'steisyen', 'automotive': 'otomotif', 'robotics': 'robotiks',
    'product': 'prodak', 'design': 'disain', 'business': 'bisnis', 'computer': 'kompyuter',
    'science': 'saiens', 'mechanical': 'mekanikal', 'electrical': 'elektrikal', 'machine': 'masyin',
    'artificial': 'artifisyal', 'intelligence': 'intelijens', 'manufacturing': 'menyufekcering',
    'technology': 'teknoloji', 'information': 'informeisyen', 'systems': 'sistems', 'supply': 'saplai',
    'chain': 'cein', 'scholarship': 'skolarsyip', 'head': 'hed',
    'development': 'divelopmen', 'fee': 'fi', 'fixed': 'fikst', 'tuition': 'tuisyen',
    'variable': 'veriabel', 'equipment': 'ekuipmen', 'expression': 'ekspresyen', 'technique': 'teknik',
    'physics': 'fisiks', 'ergonomic': 'ergonomik',
    'library': 'laibreri', 'canteen': 'kentin', 'club': 'klab', 'drone': 'dron', 'campus': 'kempus',
    'hiring': 'hairing', 'online': 'onlain', 'interview': 'interviu', 'prototype': 'prototaip',
    'prototyping': 'prototaiping', 'sketching': 'skecing', 'modelling': 'modeling', 'gadget': 'gejet',
    'tools': 'tuls', 'hub': 'hab', 'website': 'websait', 'python': 'paiton', 'c++': 'si plas plas',
    'iot': 'ai o ti', 'ai': 'e ai', 'autocad': 'oto ked', 'shell': 'syel', 'eco-marathon': 'iko maraton',
    'united': 'yunaited', 'bike': 'baik',
}
ID_RESPELL_RE = re.compile(
    r'(?<!\w)(' + '|'.join(map(re.escape, sorted(ID_RESPELL, key=len, reverse=True))) + r')(?!\w)', re.I)


def respell_id(text: str) -> str:
    return ID_RESPELL_RE.sub(lambda match: ID_RESPELL[match[0].lower()], text)


class Speech:
    def __init__(self, settings: Settings, glossary: Glossary):
        self.settings, self.glossary = settings, glossary
        self.whisper = None
        self.voices = {}

    @staticmethod
    def play(audio_path: Path) -> None:
        """Use the desktop's default output, falling back to ALSA on headless installs."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f'Audio file not found: {audio_path}')
        player = next(filter(None, map(shutil.which, ('paplay', 'pw-play', 'aplay'))), None)
        if not player:
            raise RuntimeError('No audio player found. Install alsa-utils or pulseaudio-utils.')
        result = subprocess.run([player, str(audio_path)], capture_output=True, text=True)
        if result.returncode:
            detail = result.stderr.strip()[:400] or f'exit status {result.returncode}'
            raise RuntimeError(f'{Path(player).name} playback failed: {detail}. Check the selected output in Ubuntu Sound settings.')

    def load_whisper(self):
        if self.whisper is None:
            import ctranslate2
            from faster_whisper import WhisperModel
            device = self.settings.speech_device
            if device == 'auto':
                device = 'cuda' if ctranslate2.get_cuda_device_count() > 0 else 'cpu'
            model = self.settings.whisper_model
            local = self.settings.voices_dir.parent / 'whisper' / model
            if (local / 'model.bin').exists():
                model = str(local)
            self.whisper = WhisperModel(
                model, device=device, compute_type='float16' if device == 'cuda' else 'int8',
                cpu_threads=4, num_workers=1,
            )
        return self.whisper

    def load_voice(self, language: str):
        if language not in VOICES:
            raise ValueError('Speech output language must be id or en')
        if language not in self.voices:
            from piper import PiperVoice
            model = self.settings.voices_dir / (VOICES[language] + '.onnx')
            if not model.exists() or not Path(str(model) + '.json').exists():
                raise FileNotFoundError('Piper voice missing. Run python scripts/download_models.py.')
            self.voices[language] = PiperVoice.load(str(model), use_cuda=False)
        return self.voices[language]

    def transcribe(self, audio_path: Path, language: str | None = None) -> tuple[str, str]:
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f'Audio file not found: {audio_path}')
        segments, info = self.load_whisper().transcribe(
            str(audio_path), language=language, beam_size=5, temperature=0.0,
            vad_filter=True, vad_parameters={'min_silence_duration_ms': 300},
            condition_on_previous_text=False,
            initial_prompt=self.glossary.initial_prompt, hotwords=self.glossary.hotwords,
        )
        text = ' '.join(segment.text.strip() for segment in segments if segment.no_speech_prob < 0.6).strip()
        if not text:
            raise ValueError('No speech detected; try a clearer recording or --language id/en.')
        return self.glossary.normalize_speech(text), info.language if info.language in VOICES else 'en'

    def synthesize(self, text: str, output: Path, language: str = 'en', tuning: dict | None = None) -> Path:
        from piper import SynthesisConfig
        voice = self.load_voice(language)
        tuning = dict(VOICE_TUNING.get(language, {}) if tuning is None else tuning)
        pause = tuning.pop('pause', 0.0)
        # Citations and Markdown belong in text, not in the spoken answer.
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'[*#`]', '', text)
        text = self.glossary.pronounce(text.strip(), language)
        if language == 'id':
            text = respell_id(text)
        if not text:
            raise ValueError('No text to speak')
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), 'wb') as wav:
            # Piper yields one chunk per sentence; synthesize_wav would join them with no gap.
            for index, chunk in enumerate(voice.synthesize(text, syn_config=SynthesisConfig(**tuning))):
                if index == 0:
                    wav.setframerate(chunk.sample_rate)
                    wav.setsampwidth(chunk.sample_width)
                    wav.setnchannels(chunk.sample_channels)
                else:
                    wav.writeframes(bytes(chunk.sample_width * int(chunk.sample_rate * pause)))
                wav.writeframes(chunk.audio_int16_bytes)
        return output
