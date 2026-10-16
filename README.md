# MIRAI — offline bilingual voice assistant for BINUS ASO

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/LLM-Qwen3%204B%20via%20Ollama-black)
![Hardware](https://img.shields.io/badge/Runs%20on-NVIDIA%20Jetson-76B900?logo=nvidia&logoColor=white)
![Tests](https://img.shields.io/badge/tests-90%2B%20pytest-2ea44f)

MIRAI is the campus guide for BINUS ASO School of Engineering's robot. It was designed and
built by Clayvent Yuuki William as a BCore AI Research Club project. Visitors walk up to its touchscreen, pick Bahasa Indonesia or
English, and **ask a question out loud**. MIRAI transcribes it, looks up the answer in
the university's official FAQ PDFs, and **speaks the answer back** — with everything,
including the LLM, running **locally on an NVIDIA Jetson**: no cloud API and no
Internet needed once installed.

## How it works

```text
 USB mic ──► faster-whisper (STT) ──► glossary fixes ("ukam" → UKM, "the base" → D'BASE)
                                              │
                                              ▼
 FAQ PDFs ──► FAQ-aware chunking ──► multilingual-e5 embeddings (ONNX, cached)
                                              │  semantic search + keyword bonus,
                                              ▼  calibrated cut-off
                     Qwen3 4B Instruct (Ollama) — answers only from retrieved text,
                     cites file + page, short apology when the answer isn't there
                                              │
                                              ▼
 Speaker ◄── Piper TTS (Indonesian / English voice) ◄── touchscreen kiosk (FastAPI + HTML/JS)
```

## Highlights

- **Grounded answers:** retrieval-augmented generation over the official FAQ; switching to multilingual-e5-small raised top-3 retrieval accuracy from 89% to 98% on a bilingual question set, with file and page citations. Unsupported university questions get an honest "I'm not sure, please ask nearby staff" rather than a guess.
- **Bilingual speech, offline:** faster-whisper speech-to-text and Piper text-to-speech for Indonesian and English. The Indonesian voice was tuned by ear on the device.
- **Fixing errors from real use:** a shared `glossary.json` corrects how Whisper mishears campus names and abbreviations (UKM, HMB, D'BASE, AeroBASE). The entries come from microphone tests on the actual kiosk.
- **Conversation memory per visitor:** follow-ups such as "that program" or "di sana" resolve to the previous topic. Memory is in-memory SQLite and is wiped when a visitor taps *Finish* or after 2 minutes idle, so the next person starts fresh.
- **Built for limited hardware:** quantized ONNX embeddings (~640 MiB vs ~810 MiB for PyTorch, with 0.99+ cosine agreement), a low-memory launcher, recovery when Ollama runs out of memory, and models kept loaded so the first visitor doesn't wait.
- **Kiosk deployment:** starts automatically at login, waits for the USB microphone before opening the browser, and has a cancel button while it listens, thinks or speaks.
- **Tested:** 90+ pytest tests (conversation flow, follow-ups, visitor isolation, FAQ chunking, server API, speech playback), plus evaluation scripts for retrieval (~60 bilingual questions), conversations and speech recognition.

## Tech stack

Python · Ollama (Qwen3 4B Instruct, Q4_K_M) · multilingual-e5 / ONNX Runtime · faster-whisper · Piper TTS · FastAPI · SQLite · pytest · NVIDIA Jetson (Ubuntu)

---

## Setup and operations

Rebuilt from [`Skyresist/BASE-Core-AI-LLM`](https://github.com/Skyresist/BASE-Core-AI-LLM), the first
text-only prototype. MIRAI answers questions from local PDFs, remembers the current
visitor's conversation, and accepts spoken questions and speaks answers in Indonesian
or English. No paid API key is needed.

The default model is **Qwen3 4B Instruct 2507, Q4_K_M**, through Ollama:
`qwen3:4b-instruct-2507-q4_K_M`. This avoids the moving `qwen3:4b` tag's Thinking
variant and its extra latency. Inference is local after model downloads.

## Fresh Jetson install

See [JETSON_START_HERE.md](JETSON_START_HERE.md) for transferring the ZIP from
your laptop. Inside the extracted project on the Jetson:

```bash
bash scripts/install_jetson.sh
```

The installer prepares Ubuntu prerequisites, `.venv`, CPU-compatible Python
packages, Ollama 0.35.1 if absent, Qwen, multilingual embeddings, faster-whisper
`base`, and both Piper voices. It then runs tests and the functional smoke check.
It supports Ubuntu 22.04/JetPack 6 and Ubuntu 24.04 with systemd, and needs a normal user with sudo and
Internet access. Allow about 6 GB of downloads and at least 15 GB free space.
Model weights are not included in the source ZIP.

CPU speech is the first-run default. Ollama detects supported GPUs.
`--no-system-packages` skips apt installation. Existing PyTorch in `.venv` is
preserved; this script does not flash JetPack or replace NVIDIA drivers.

On Ubuntu 22.04, the default Python 3.10 is too old for this application. The
installer uses a private uv-managed Python 3.12 under `.python/`, with bootstrap
tools in `.setup-tools/`. System Python and NVIDIA packages stay in place. An
existing incompatible `.venv` is preserved under `.venv-backup.*/venv` before
creating the new environment. Keep `.python/` alongside `.venv` when moving this
installed folder: the environment refers to that interpreter.

## Touchscreen kiosk (robot screen)

The robot's touchscreen runs a full-screen web page served by `server.py` on the
Jetson itself. Visitors pick Bahasa Indonesia or English, then tap the microphone
and speak (recording stops automatically after they finish), tap a suggested
question, or type. Answers appear on screen and play through the Jetson's speaker.

```bash
bash scripts/start_kiosk.sh                # server + Chromium kiosk, 8 GB profile
bash scripts/start_kiosk.sh --server-only  # then open http://127.0.0.1:8000
```

- The UI is plain HTML/JS in `static/` with no build step and no Internet assets.
- `localhost` is a secure browser origin, so the USB microphone works without HTTPS.
  Select the USB mic/speaker as Ubuntu's default input/output in Sound settings.
- "Selesai/Finish" or 120 idle seconds (`--idle-timeout`) clears the visitor and
  returns to the language screen; a warning shows 15 seconds before.
- Qwen stays loaded (`keep_alive=-1`) and Whisper/Piper load at startup, so the
  first visitor does not wait for model loading.
- API: `POST /api/ask {text, language}`, `POST /api/transcribe` (multipart audio),
  `POST /api/speak {text, language}` → WAV, `POST /api/reset`, `GET /api/status`.
  The ROS side can use the same API later.

`python scripts/eval_retrieval.py --backend onnx` checks that ~60 English and
Indonesian visitor questions retrieve the right FAQ entry. Run it after editing
the FAQ PDF, `glossary.json` or retrieval settings.

## Real university knowledge

Put your current BINUS ASO PDFs in `data/pdfs/`, including subdirectories. The
original repository contained no PDFs; this bundle contains no invented campus
knowledge. Sources show filenames and pages, and low-scoring matches are omitted.
Unsupported university questions get an explicit uncertainty response.

```bash
bash scripts/start.sh
bash scripts/start.sh -q "Apa saja UKM di BINUS ASO?"
bash scripts/start.sh -q "Apa saja UKM?" --retrieve-only
```

The built-in loader needs a PDF text layer. It reports scanned pages requiring
OCR; PaperRAG is an optional path for more complex document parsing.

## Memory and cache

"Who are you?", "Siapa kamu?" and related introduction questions give a short
MIRAI introduction in the selected language. This uses a fixed self-description
without searching PDFs or calling Ollama; it is saved only in the current visit.

Follow-up retrieval uses the latest real question and answer to interpret phrases
such as "that program", "which one", "di sana" and "program itu". It does not
accumulate older program descriptions into later searches. The same recent
reference is supplied beside the current question to help Qwen keep the selected
program. Earlier messages identify the topic; new university facts still need
PDF evidence. A dangling reference with no topic asks a short clarification.
Explicit new subjects start their own query.

Unsupported university questions use a short apology and suggest approaching
nearby staff. University terms or a university follow-up identify that route;
unrelated unknown questions keep the general apology. This is lightweight
English/Indonesian pattern matching, not a universal intent classifier. Context
comes only from the current visitor's bounded history and is cleared on reset.

- Full conversation turns stay in an in-memory SQLite database for the current visit.
  Each launch starts fresh, even with the same `--session` label. Old saved
  `.mirai/memory.sqlite3` files are ignored; they are not automatically deleted.
- A bounded window of the latest six turns enters the prompt. Ollama can call
  `recall_memory` for older messages and `search_documents` for PDF facts.
- Type `done`, `finish`, `selesai`, `reset` or `/reset` to clear the conversation
  immediately for a new visitor, without confirmation or stopping MIRAI.
  `/done`, `exit`, `bye` and `keluar` also clear the visitor and keep it running.
  After each reply the text prompt offers another question or `done` / `finish`.
  In a Linux terminal, five minutes without a submitted line clears the visitor
  and returns to `You:`. Waiting without an active visitor does not repeat the
  reset message. PDF retrieval and speech components are reused across visitors.
  Use `/exit` or Ctrl+C to shut down MIRAI; EOF and errors also close its memory.
  Set `--idle-timeout 120` for two minutes. Generation/playback time is excluded.
- Embeddings persist in `.mirai/embeddings.npz`. Document text, page metadata,
  model name and model revision determine cache validity. Changed documents
  rebuild the cache; unchanged documents do not get re-embedded.
- Conversation memory supports names/preferences/follow-ups; it is not evidence
  of university facts. The implementation uses recent history plus search,
  rather than automatic generated summaries.
- Tool rounds, total calls and total tool-output size are bounded.

```bash
bash scripts/start.sh --idle-timeout 300
# In this same chat: Nama saya Raka -> Ingat nama saya? -> finish
# Ask about your name again: the previous visitor is no longer known.
```

## Speech and Indonesian terminology

```bash
bash scripts/start.sh --audio question.wav --language id --speak answer.wav
```

Use `--language en` for English; `auto` uses Whisper's detected audio language.
The start script plays typed and transcribed answers through Ubuntu's default
sound output, using paplay, pw-play or aplay. Select your speaker in Ubuntu Sound
settings. Use `bash scripts/start.sh --no-play` for silent text, or add `--no-play`
to `--speak answer.wav` to save without playback. `python app.py` keeps playback
opt-in with `--play`. Without `--speak`, spoken answers use `.mirai/last-answer.wav`
(or your configured state directory). Each new answer overwrites that WAV.
Speech errors are reported while the text conversation continues.
Input supports formats PyAV can decode. The command line accepts audio files;
live microphone input is handled by the touchscreen kiosk.

Replies default to 1–2 short sentences (aiming for under 60 words) and a 192-token generation budget to reduce
waiting and spoken-answer length. For detailed replies, ask for detail and launch
with `--max-tokens 512`. The first model load may still take longer. Run `ollama ps`
just after a reply to see the CPU/GPU allocation; do not infer GPU use from the
presence of CUDA libraries alone.
MIRAI uses PDFs to verify facts and explains them in its own conversational prose,
preserving proper names and important numbers. Unsupported answers use a short
formal apology without narrating missing PDFs. Bullets are reserved for explicit
list requests. Text citations remain available; speech omits citations.
The default context window is 4096 tokens with smaller recent-history and PDF
prompt budgets to reduce memory consumption. Full conversation history remains
in memory during this visit and can be searched by the memory tool. Use `--num-ctx 8192` to restore
the larger Ollama context allocation if needed. GPU offloading still depends on
the board, available memory and other processes.

For silent output, run `bash scripts/check_audio.sh --play` and inspect the default
sink, mute/volume, audio hardware and saved WAV. This script does not change your
audio settings. Sound plays through a speaker/headset connected to the Jetson.

- ASR: faster-whisper `small` (beam 1, 6 threads), CPU int8, VAD enabled, previous-context conditioning
  disabled, and university vocabulary supplied as initial-prompt/hotword hints.
- Indonesian Piper voice: `id_ID-news_tts-medium`.
- English Piper voice: `en_US-lessac-medium`.
- Piper uses local ONNX/VITS voices with its bundled eSpeak phonemizer; a separate
  VITS server is unnecessary for these voices.

Edit [`glossary.json`](glossary.json) on your laptop to add real misheard spellings
and pronunciation preferences. It is shared by ASR correction, search expansion
and TTS. UKM, BINUS ASO, D'BASE and AeroBASE are included. Aliases match whole
words. The glossary guides searches; it does not establish facts about clubs.
Hints help but cannot guarantee accuracy on noisy recordings.

If the Jetson runs short of memory, `--whisper-model base` is about twice as fast
but mishears far more (38% vs 20% word errors), or supply a downloaded CTranslate2
model directory.
An explicit `--language id` often helps campus questions.

## CUDA and TensorRT on Jetson

Check the actual board and installed platform first:

```bash
cat /etc/nv_tegra_release
cat /etc/os-release
tr -d '\0' </proc/device-tree/model
ollama ps
```

Ollama GPU inference, PyTorch embedding acceleration, and CTranslate2 ASR are
independent. JetPack 7/Ubuntu 24 alone does not identify the board or guarantee
compatible third-party wheels.

- Keep ASR on CPU until a compatible CTranslate2 build and CUDA 12/cuDNN 9 runtime
  are verified for your board, then try `--speech-device cuda`.
- `--embedding-device cuda` requires matching NVIDIA/PyTorch packages.
- A PyPI CTranslate2 wheel may need a Jetson-specific build for CUDA. Never copy
  x86 wheels to ARM64.
- TensorRT optimizes supported networks for particular NVIDIA hardware. It is
  not a replacement for CUDA or an automatic switch for Ollama/faster-whisper.
  TensorRT-LLM or Whisper engines would be a separate deployment after measuring
  the working baseline and confirming the exact board and JetPack version.

## 8 GB Jetson memory profile

If Linux logs `Out of memory: Killed process`, close heavy desktop applications
on the Jetson and use the laptop for editing/chat. Exit any other MIRAI process,
then launch:

```bash
bash scripts/start_low_memory.sh
```

This keeps Qwen 4B, PDF citations, memory during the visit and spoken answers.
It uses the official ARM64 INT8 ONNX export of the same multilingual embedding
model, a 3072-token Ollama context, two PDF matches and a 120-token reply budget.
Semantic retrieval remains available for English questions about Indonesian PDFs.
The encoder uses ONNX Runtime and the Rust tokenizer directly, without importing
PyTorch. It retains the model's 128-token truncation, mean pooling and normalized
vectors. Quantization can cause small differences in scores or ranking.
Recent-memory and evidence prompt windows are smaller; the dense
embedding cache are preserved. ONNX uses a separate `embeddings-onnx.npz` cache.
For an existing installation, apply the source update, then run
`bash scripts/install_semantic_search.sh` once. It installs the two runtime
dependencies, downloads about 122 MiB of pinned model/tokenizer assets and
verifies a live PDF/memory workflow. The normal installer includes this setup.
The earlier TF-IDF-only low-memory profile was replaced after it missed basic
cross-language FAQ questions. Word-only search remains an explicit diagnostic
option (`--embedding-backend tfidf`), and is never selected automatically.
Use the regular `start.sh` for the original PyTorch semantic encoder.
Explicit `--num-ctx`, `--max-tokens` and `--top-k` options override this profile.
When you ask about your name, the latest saved self-introduction is recovered
inside the existing history budget. It is scoped to your session and cleared by
reset, exit or timeout; name questions do not pick up unrelated university PDF citations.

Run `bash scripts/check_memory.sh` while MIRAI is open to inspect RAM, swap,
largest process RSS and loaded Ollama models. The script does not change settings.
A nearly full machine after `ollama stop` needs this check; stopping one model
does not establish that all model runners or other processes have exited.
Extra swap can reduce some host-memory crashes but does not provide GPU memory
and can make inference slower. Do not change swap until the current devices and
remaining processes have been inspected.

## Optional PaperRAG integration

The default is a smaller local retrieval pipeline using multilingual embeddings
and lexical scores. Persistent indexing, provenance, score thresholds and the
retrieval-only workflow follow the local-first approach in
[bagustris/PaperRAG](https://github.com/bagustris/PaperRAG), without adding its
heavy document/OCR stack to the default Jetson environment.

An adapter can query an existing **trusted PaperRAG index**:

```bash
python app.py --paperrag-index /path/to/.paperrag-index -q "Apa saja UKM?"
```

PaperRAG must be importable in that Python environment. Use its own documented
`uv` setup, then install MIRAI's requirements into that environment and run
MIRAI with its interpreter. For example, in PaperRAG after its `uv sync`:

```bash
uv pip install --python .venv/bin/python -r /path/to/MIRAI/requirements.txt
uv run paperrag index --input-dir /path/to/pdfs --embed-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
.venv/bin/python /path/to/MIRAI/app.py --paperrag-index /path/to/pdfs/.paperrag-index -q "Apa saja UKM?"
```

The adapter reads `config_snapshot.json` to match the index's embedder. PaperRAG
provides section/file references rather than page labels through this adapter.
Its Docling/OCR installation is optional and separate. Only load your own
indexes: upstream metadata may use pickle.

## Development and checks

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-voice.txt -r requirements-dev.txt
# Run ollama serve in a separate terminal first.
.venv/bin/python scripts/download_models.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_test.py --live --voice
```

Public model revisions are pinned in `scripts/download_models.py`. Piper files
are checked against the publisher's manifest before installation; retain the
downloaded MODEL_CARD files and review model/dependency licenses before sharing
weights. No credentials are embedded in the source.

Behind a proxy that prevents Ollama's server-side redirects, use:

```bash
.venv/bin/python scripts/pull_model.py --registry-import
```

This downloads the official manifest's weight/template/license/parameter layers,
verifies SHA256 and size, imports them using `ollama create`, then deletes
temporary downloads. TLS/checksum verification stays enabled. An unrestricted
Jetson connection should not need this fallback.

Ordinary tests require no downloaded models. The smoke script creates temporary
test PDFs and checks actual retrieval plus optional Qwen generation, memory during
the same visit and fresh-visitor isolation, real tool calls, both voices,
transcription and silence rejection. It
never puts synthetic campus facts into `data/`.

`--embedding-backend tfidf` is an explicit lexical alternative for offline
diagnostics, without multilingual semantic matching. The regular launcher uses
PyTorch dense retrieval; the low-memory launcher uses quantized ONNX dense
retrieval. Empty collections do not download an embedding model.

Check Ollama using `curl http://127.0.0.1:11434/api/version` and `ollama list`.
Use `--verbose` for loading details, or `sudo journalctl -u ollama -n 40` for the
Jetson service. Where Xet cannot work through a proxy, Hugging Face's supported
HTTP download route can be selected with `HF_HUB_DISABLE_XET=1`.

The original novelty response for “who is the number one idol” remains in
`secret.py`; that file contains artwork, not credentials.
