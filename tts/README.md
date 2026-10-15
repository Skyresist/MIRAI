# MIRAI RVC TTS (experimental, standalone)

Text → [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS → [Applio](https://github.com/IAHispano/Applio) RVC voice conversion → wav.

## Setup

Standalone tool, not wired into MIRAI's speech yet. Needs x86-64 Linux, Python 3.12, git, ffmpeg, and an NVIDIA GPU (falls back to CPU). Takes ~9 GB.

```bash
./setup.sh
```

The first run also downloads the RMVPE/ContentVec (~600 MB) and Kokoro (~330 MB) models.

## Usage

Put an RVC v2 `.pth` (and its `.index`, same name) in this folder. Models are not included.

```bash
./speak.py -m model.pth "Hello everyone, welcome back to the stream!" -o out.wav
./speak.py -m model.pth -f script.txt -o out.wav
```

| Option | Default | Notes |
|---|---|---|
| `--pitch` | 6 | Semitones. Kokoro's `af_*` voices are fairly low for anime-style models |
| `--protect` | 0.5 | Keeps consonants clean (fixes words like "Minecraft") |
| `--index-rate` | 0.75 | Lower if robotic, higher for more timbre match |
| `-v` | af_heart | Any Kokoro voice, e.g. `af_bella`, `am_adam` |
| `--keep-tts` | off | Also saves the raw Kokoro audio |

Only use voice models you have the right to use, and don't present output as a real person actually speaking.
