from pathlib import Path
from subprocess import CompletedProcess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import pytest

import app
import speech
from llm import OllamaChat
from speech import Speech


@pytest.mark.parametrize('available', ['paplay', 'pw-play', 'aplay'])
def test_playback_uses_available_player_without_shell(tmp_path, monkeypatch, available):
    # Spaces in user paths must remain one argument to the system player.
    audio = tmp_path / 'spoken answer.wav'
    audio.write_bytes(b'placeholder')
    monkeypatch.setattr(speech.shutil, 'which', lambda name: f'/usr/bin/{name}' if name == available else None)
    run = Mock(return_value=CompletedProcess([], 0, '', ''))
    monkeypatch.setattr(speech.subprocess, 'run', run)
    Speech.play(audio)
    run.assert_called_once_with([f'/usr/bin/{available}', str(audio)], capture_output=True, text=True)


def test_player_failure_explains_sound_output_problem(tmp_path, monkeypatch):
    audio = tmp_path / 'answer.wav'
    audio.touch()
    monkeypatch.setattr(speech.shutil, 'which', lambda name: '/usr/bin/paplay' if name == 'paplay' else None)
    monkeypatch.setattr(speech.subprocess, 'run', Mock(return_value=CompletedProcess([], 1, '', 'Connection refused')))
    with pytest.raises(RuntimeError, match='paplay playback failed: Connection refused'):
        Speech.play(audio)


def test_speech_failure_keeps_bilingual_text_conversation_running(tmp_path, monkeypatch, capsys):
    voice = Mock()
    voice.play.side_effect = RuntimeError('No speaker connected')
    monkeypatch.setattr(speech, 'Speech', lambda settings, glossary: voice)
    monkeypatch.setattr(OllamaChat, 'answer', lambda *args, **kwargs: 'Hello from MIRAI.')
    questions = iter(['Hello', 'Halo', '/exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(questions))
    assert app.main(['--pdf-dir', str(tmp_path / 'pdfs'), '--state-dir', str(tmp_path / 'state'),
                     '--embedding-backend', 'tfidf', '--play']) == 0
    assert voice.synthesize.call_count == 2
    assert [call.args[2] for call in voice.synthesize.call_args_list] == ['en', 'id']
    assert voice.synthesize.call_args.args[1] == tmp_path / 'state' / 'last-answer.wav'
    assert voice.play.call_count == 2
    output = capsys.readouterr()
    assert output.out.count('Bot: Hello from MIRAI.') == 2
    assert output.err.count('Speech unavailable: No speaker connected') == 2


def test_no_play_can_save_wave_without_accessing_sound_device(tmp_path, monkeypatch):
    voice = Mock()
    monkeypatch.setattr(speech, 'Speech', lambda settings, glossary: voice)
    monkeypatch.setattr(OllamaChat, 'answer', lambda *args, **kwargs: 'Hello.')
    output = tmp_path / 'answer.wav'
    assert app.main(['-q', 'Hello', '--pdf-dir', str(tmp_path / 'pdfs'),
                     '--state-dir', str(tmp_path / 'state'), '--embedding-backend', 'tfidf',
                     '--play', '--no-play', '--speak', str(output)]) == 0
    voice.synthesize.assert_called_once_with('Hello.', output, 'en')
    voice.play.assert_not_called()


def test_sentences_get_a_breathing_pause_and_language_tuning(tmp_path, monkeypatch):
    # Fake Piper: two one-sample sentences; records the config it was given.
    chunk = SimpleNamespace(sample_rate=10, sample_width=2, sample_channels=1, audio_int16_bytes=b'\x01\x00')
    voice = Mock(synthesize=Mock(return_value=[chunk, chunk]))
    monkeypatch.setitem(sys.modules, 'piper', SimpleNamespace(SynthesisConfig=dict))
    talker = Speech(Mock(), Mock(pronounce=lambda text, language: text))
    talker.voices = {'id': voice, 'en': voice}
    talker.synthesize('Satu. Dua.', tmp_path / 'id.wav', 'id')
    assert voice.synthesize.call_args.kwargs['syn_config'] == {'length_scale': 1.1, 'noise_scale': 1.0, 'noise_w_scale': 0.9}
    with wave.open(str(tmp_path / 'id.wav')) as wav:
        assert wav.getnframes() == 1 + 3 + 1  # 0.3 s of silence at 10 Hz between the sentences
    talker.synthesize('One. Two.', tmp_path / 'en.wav', 'en')
    assert voice.synthesize.call_args.kwargs['syn_config'] == {}
    with wave.open(str(tmp_path / 'en.wav')) as wav:
        assert wav.getnframes() == 2  # English unchanged: no added pause


@pytest.mark.parametrize('heard, expected', [
    ('Singkatan kampus BINUS ASO yaitu UKM dan HMB dan TPKS.', None),  # Whisper copied its hint.
    ('Singkatan kampus ASO yaitu UKM dan HMB.', None),
    ('Saya mau tanya tentang UKM', 'Saya mau tanya tentang UKM'),
    ('HMB itu apa?', 'HMB itu apa?'),
])
def test_a_copied_hint_counts_as_no_speech(heard, expected):
    from glossary import Glossary
    segment = SimpleNamespace(text=heard, no_speech_prob=0.1)
    talker = Speech(SimpleNamespace(whisper_beam=1), Glossary())
    talker.whisper = Mock(transcribe=Mock(return_value=([segment], SimpleNamespace(language='id'))))
    audio = Path(__file__)  # Any existing file; the fake Whisper ignores it.
    if expected is None:
        with pytest.raises(ValueError):
            talker.transcribe(audio, 'id')
    else:
        assert talker.transcribe(audio, 'id')[0] == expected


def test_whisper_loops_are_collapsed():
    assert speech.collapse_repeats('Ada UKM apa saja? BINUS ASO, BASE BASE BASE') == 'Ada UKM apa saja? BINUS ASO, BASE'
    assert speech.collapse_repeats('BINUS ASO, BINUS ASO, BINUS ASO.') == 'BINUS ASO,'
    assert speech.collapse_repeats('Apa itu BINUS ASO?') == 'Apa itu BINUS ASO?'


def test_english_terms_are_respelled_for_the_indonesian_voice_only(tmp_path, monkeypatch):
    assert speech.respell_id('Jurusan Automotive & Robotics Engineering, pakai C++ dan AI.') == \
        'Jurusan otomotif en robotiks enjiniring, pakai si plas plas dan e ai.'
    assert speech.respell_id('Program project-based learning') == 'Program projek beist lerning'
    assert speech.respell_id('Ainun belajar di kampus') == 'Ainun belajar di kampus'  # whole words only
    chunk = SimpleNamespace(sample_rate=10, sample_width=2, sample_channels=1, audio_int16_bytes=b'\x01\x00')
    voice = Mock(synthesize=Mock(return_value=[chunk]))
    monkeypatch.setitem(sys.modules, 'piper', SimpleNamespace(SynthesisConfig=dict))
    talker = Speech(Mock(), Mock(pronounce=lambda text, language: text))
    talker.voices = {'id': voice, 'en': voice}
    talker.synthesize('Business Engineering', tmp_path / 'en.wav', 'en')
    assert voice.synthesize.call_args.args[0] == 'Business Engineering'
    talker.synthesize('Business Engineering', tmp_path / 'id.wav', 'id')
    assert voice.synthesize.call_args.args[0] == 'bisnis enjiniring'
