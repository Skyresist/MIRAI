# Start MIRAI on your new Jetson

1. Download the source ZIP from this chat to your laptop. Models and university
   PDFs are downloaded/added separately.
2. Extract/copy it to the Jetson using USB or an already working SSH connection.
   You can type SSH commands on the laptop without a keyboard on the Jetson.
3. Inside the extracted project on the Jetson, run:

   ```bash
   bash scripts/install_jetson.sh
   ```

4. Copy current BINUS ASO PDFs into `data/pdfs/`.
5. Run:

   ```bash
   bash scripts/start.sh
   ```

Example transfer (replace username and IP with the actual Jetson values):

```bash
scp MIRAI-Jetson.zip YOUR_USER@JETSON_IP:~/
ssh -t YOUR_USER@JETSON_IP
# These commands now run on the Jetson:
unzip -n MIRAI-Jetson.zip
cd MIRAI-Jetson
bash scripts/install_jetson.sh
bash scripts/start.sh
```

SSH must already be enabled and reachable. If `unzip` is missing, use
`sudo apt-get install unzip`, or extract the folder on the laptop before copying.
The first setup needs Internet and sudo for Ubuntu packages/the Ollama service.
No cloud API key is needed. See README for space requirements and hardware limits.

Ask once:

```bash
bash scripts/start.sh -q "Apa saja UKM di BINUS ASO?"
```

Test memory:

```bash
bash scripts/start.sh
# In the same chat: Nama saya Raka -> Ingat nama saya? -> finish
# Ask your name again: it must not know the previous visitor.
```

Use a recorded question and generate a spoken answer:

```bash
bash scripts/start.sh --audio question.wav --language id --speak answer.wav
```

The start script now plays answers aloud through the default Ubuntu sound output.
Choose your speaker in Ubuntu Sound settings. For silent text, use
`bash scripts/start.sh --no-play`. Speech failures print an error and keep the text
chat running. The latest spoken answer is saved as `.mirai/last-answer.wav` unless
you supply `--speak`.

Replies are brief by default. For longer explanations, ask for detail and run
`bash scripts/start.sh --max-tokens 512`.

On an 8 GB Jetson with out-of-memory crashes, close other MIRAI instances and
launch `bash scripts/start_low_memory.sh`. It keeps Qwen 4B, spoken answers and
conversation memory during the visit while using the same multilingual model's quantized
ONNX semantic embeddings and smaller prompt windows. For an existing install,
run `bash scripts/install_semantic_search.sh` once after applying the source
update. The earlier word-only profile has been replaced. Run
`bash scripts/check_memory.sh` to inspect memory and loaded processes/models.

The first voice interface uses audio files, not live microphone recording.
Edit `glossary.json` on your laptop to add observed speech mistakes and preferred
pronunciations. `--language id` helps Indonesian campus questions.

Your exact Jetson board is still unknown. Start with CPU Python; Ollama detects
supported GPUs. Do not install unrelated CUDA/TensorRT packages before identifying
the board. The README explains the next acceleration checks.

Ubuntu 22.04 / JetPack 6 is supported by the updated installer. Its Python 3.10
stays as the system interpreter; MIRAI gets a private Python 3.12. Keep `.python/`
and `.setup-tools/` with your installed project. Any incompatible previous
environment is preserved in `.venv-backup.*/venv`.

Keep `.mirai/` and `models/` when updating code to preserve search caches and weights.
Conversation memory is held only in RAM. `done`, `finish`, `selesai`, `reset`,
`/done` and ordinary `exit` clear the visitor and return to `You:` immediately.
Use `/exit` or Ctrl+C to stop MIRAI; EOF or application errors also close it.
The old `memory.sqlite3` archive is ignored and left untouched.
Five idle minutes clear the visitor and return to `You:` without stopping MIRAI. Use
`--idle-timeout 120` for two minutes; model generation and audio playback do not
count as idle time. The timeout does not apply to redirected stdin or one-shot requests.
