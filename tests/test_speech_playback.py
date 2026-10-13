from subprocess import CompletedProcess
from unittest.mock import Mock

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
