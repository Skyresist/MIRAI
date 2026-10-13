# Validation — 12 October 2026

Validated on the cloud's x86_64 Linux CPU machine with Python 3.12.14 and Ollama
0.35.1. This is not a claim that installation or GPU inference has been tested
on the user's Jetson.

- 91 repository tests passed; dependency consistency and shell syntax checks passed.
- English and Indonesian self-introductions use a deterministic MIRAI response,
  without PDF retrieval or Ollama generation. Tests cover identity wording,
  explicit language selection, saved visitor-name preservation, reset and
  keeping combined university questions on the normal retrieval path.
- The follow-up update passed real ONNX/Qwen English and Indonesian three-turn
  chains using temporary synthetic PDFs: list programs, select the one involving
  cars, then retrieve the selected program's lab. Both used Vehicle Dynamics Lab
  for the synthetic Automotive Engineering program and did not switch to the
  synthetic Robotics Engineering program's Control Systems Lab. An initial live
  failure exposed older query accumulation and an ambiguous program reference;
  the final code uses only the latest pair for retrieval and supplies that same
  reference beside the current question. Live unsupported university questions
  redirected to nearby staff; English/Indonesian quantum-physics questions kept
  the general apology. Tests cover subject changes, personal-memory isolation,
  dangling references, context bounds and clearing topic context between visitors.
  University detection uses language patterns, not an exhaustive semantic model.
  The user's actual FAQ remains on the Jetson and needs the same checks there.
- Visitor sessions now use in-memory SQLite. Tests cover fresh launches with an
  old archive present, immediate `done`/`finish`/`selesai` resets, `/exit`,
  EOF, Ctrl+C, generation errors and idle expiry including a real terminal
  subprocess. Finish and timeout clear names/history, keep MIRAI running and
  allow the next visitor to chat in the same process. Quiet idle periods do not
  repeat reset messages. Actual process exits close visitor memory; search caches
  remain separate. The earlier confirmation-and-exit behavior was replaced.
  The live ONNX/Qwen smoke passed within-visit recent/older-name recall, fresh
  visitor isolation, PDF grounding and a real Python tool round trip.
  Live English and Indonesian unsupported 2030-fee questions returned the exact
  short apology, even with unrelated 2025-fee evidence. Earlier restart-memory
  checks below describe the previous behavior; persistence is now opt-in only
  through explicit file-backed Memory library instances, never the default CLI.
- Real PDF extraction, dense multilingual retrieval, page references and embedding
  cache reuse passed. Embeddings loaded successfully with Internet disabled.
- Qwen3 4B Instruct answered from a temporary PDF, remembered a name after an
  assistant restart, and completed a real Ollama Python memory-tool round trip.
- Indonesian and English Piper voices generated playable WAV files. Faster-whisper
  base correctly recognized BINUS ASO in both generated recordings and UKM in
  the Indonesian recording. Silence produced no fabricated question.
- After the short-reply/playback update, live Qwen grounding, name recall, tool
  calls and both voice round trips passed again with the 192-token default.
  Playback tests covered system player selection, error reporting, silent WAV
  output and continued bilingual text chat after a speaker failure. Physical
  speaker playback on the Jetson remains unverified.
- The 4096-token context update passed all 22 tests and live PDF grounding, name
  recall and memory-tool checks. The audio diagnostic recognized a real Piper
  WAV's duration and nonzero samples and reported the cloud's missing audio
  player clearly. Jetson output-device and mute diagnosis requires its output.
- The earlier lexical low-memory launcher passed actual PDF/Qwen responses, name memory across
  restarts and Indonesian/English Piper WAV generation with a 2048-token context.
  Tests reject dense/PaperRAG search in that profile and ensure lexical PDF search
  and saved-memory recall work with heavy embedding imports blocked.
- In separate cloud CPU processes querying one synthetic document, TF-IDF
  retrieval peaked at 109.6 MiB RSS and dense retrieval at 807.1 MiB RSS. These
  figures exclude Ollama and vary by platform and document collection. The actual
  low-memory CLI processes including Piper peaked at 399.9 MiB RSS on this cloud.
- The current low-memory launcher restores semantic search using the official
  quantized ARM64 ONNX export of the same pinned multilingual embedding model.
  The source export is used unchanged. It passed the actual English campus
  definition, Indonesian definition and facility-question CLI paths with Piper
  WAV output and model downloads disabled. The one-command update helper passed
  dependency checks, download/cache setup, live grounding, recent/older-name
  recall and a real Python tool round trip.
- On six English/Indonesian test queries, float-versus-quantized embedding cosine
  similarity ranged from 0.9909 to 0.9964. Separate-process peak RSS was 640.9 MiB
  for ONNX and 809.3 MiB for PyTorch; ONNX imported neither PyTorch nor
  SentenceTransformer. The actual ONNX CLI with Piper peaked at 798.5 MiB.
  These figures exclude Ollama, are measured on x86_64 cloud CPU, and do not
  establish Jetson RAM usage or eliminate the need to close heavy desktop apps.
  ARM64/Python 3.12 ONNX Runtime 1.30.0 wheels are published for manylinux 2.28,
  compatible with Ubuntu 22.04's glibc. Jetson execution remains to be checked.
- Both languages passed live conversational-style checks using a temporary test
  PDF: concise prose, fresh everyday wording, source citation and a Piper WAV.
  Initial formal wording was observed and corrected by placing a conversational
  response instruction beside the question, in addition to the system prompt.
- Personal-name questions skip unrelated PDF matches. A memory-specific prompt
  and bounded recovery of the latest saved self-introduction keep names available
  after older turns leave the recent-history window. Tests cover the prompt
  budget, renaming, reset/session isolation and continued PDF use for university
  names. Live older-name recall and the real Python memory-tool round trip passed.
- The actual command-line path accepted a spoken Indonesian question, returned
  the correct test-PDF answer with citation, and wrote a spoken answer, offline.
- The PaperRAG adapter read an actual saved FAISS index and returned a relevant
  source. Full Docling parsing/OCR was not installed or validated.
- The reusable cloud dependency/model installation script completed successfully.
- Published ARM64/Python 3.12 wheels were found for CPU PyTorch 2.8.0, Piper 1.3.0,
  CTranslate2 4.8.2 and PyAV 16.0.1. This does not verify Jetson CUDA support.
- The Python bootstrap upgraded a separate test project's Python 3.10 environment
  to a private Python 3.12.14 environment, preserved the old environment, and
  reused the new environment on a second run. An ARM64 wheel for its pinned uv
  bootstrap tool was downloaded successfully. This was tested on x86_64 Linux.

All cloud functional checks used temporary synthetic documents outside the university
knowledge directory. The user's real PDFs remain on the Jetson and are not in
the source ZIP. The user reported successful English/Indonesian FAQ answers with
the regular semantic encoder and confirmed TTS playback. Speech checks here
used synthesized recordings, not a real Jetson microphone or noisy campus audio.

The current `qwen3:4b` registry tag resolved to a Thinking model, so the default
pins the Qwen3 4B Instruct 2507 Q4_K_M tag. The Whisper downloader was corrected
to include vocabulary.txt. PyAV 16.0.1 was pinned because version 19 is incompatible
with faster-whisper 1.2.1's audio decoder. A campus-specific ASR correction for
`d'Binus ASO` was added after an observed transcription error.

Cloud configuration fields `install_script`, `start_skill`, and required model
download domains were saved as a draft. Publication and testing a newly restored
task are separate product actions and have not been claimed here.
