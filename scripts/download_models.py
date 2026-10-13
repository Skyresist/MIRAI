"""Download public models once; subsequent text and voice inference is local."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_MODEL, ROOT, Settings

WHISPER = {
    'base': ('Systran/faster-whisper-base', 'ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66'),
    # More accurate for Indonesian; ~2x slower on CPU and ~300 MB more RAM.
    'small': ('Systran/faster-whisper-small', '536b0662742c02347bc0e980a01041f333bce120'),
}
PIPER_REPO = 'rhasspy/piper-voices'
PIPER_REVISION = 'c10ece1aade47bb51c153c893d14e5bf8e5b7117'
VOICE_NAMES = ['id_ID-news_tts-medium', 'en_US-lessac-medium']


def download_voices(destination: Path):
    from huggingface_hub import hf_hub_download
    manifest = json.loads(Path(hf_hub_download(PIPER_REPO, 'voices.json', revision=PIPER_REVISION)).read_text())
    destination.mkdir(parents=True, exist_ok=True)
    for name in VOICE_NAMES:
        for filename, metadata in manifest[name]['files'].items():
            cached = Path(hf_hub_download(PIPER_REPO, filename, revision=PIPER_REVISION))
            data = cached.read_bytes()
            if len(data) != metadata['size_bytes'] or hashlib.md5(data, usedforsecurity=False).hexdigest() != metadata['md5_digest']:
                raise RuntimeError(f'Voice checksum mismatch: {filename}; refusing to install it.')
            target = destination / (name + '_MODEL_CARD' if filename.endswith('MODEL_CARD') else Path(filename).name)
            if not target.exists() or target.read_bytes() != data:
                temporary = target.with_suffix(target.suffix + '.download')
                shutil.copyfile(cached, temporary)
                temporary.replace(target)
        print('Verified Piper voice:', name, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-ollama', action='store_true')
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--onnx-only', action='store_true', help='Download only the smaller semantic-search assets.')
    parser.add_argument('--model', default=BASE_MODEL)
    parser.add_argument('--embedding-model', default=Settings().embedding_model)
    parser.add_argument('--whisper-model', choices=sorted(WHISPER), default='base')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    args = parser.parse_args(argv)
    from embeddings import cached_asset, model_spec, onnx_files
    embed_repo, embed_revision = args.embedding_model, model_spec(args.embedding_model)['revision']
    for filename in onnx_files(embed_repo):
        cached_asset(embed_repo, filename, embed_revision)
    print('Quantized multilingual ONNX embedding model cached:', embed_repo, flush=True)
    if args.onnx_only:
        return
    from huggingface_hub import snapshot_download
    snapshot_download(embed_repo, revision=embed_revision,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'], max_workers=2)
    print('Multilingual embedding model cached.', flush=True)
    if not args.text_only:
        repo, revision = WHISPER[args.whisper_model]
        snapshot_download(repo, revision=revision,
                          local_dir=args.models_dir / 'whisper' / args.whisper_model, max_workers=2,
                          allow_patterns=['config.json', 'model.bin', 'tokenizer.json', 'vocabulary.json', 'vocabulary.txt', 'preprocessor_config.json'])
        print(f'faster-whisper {args.whisper_model} downloaded.', flush=True)
        download_voices(args.models_dir / 'voices')
    if not args.skip_ollama:
        from pull_model import pull
        pull(args.model)
        print('Ollama model ready:', args.model)


if __name__ == '__main__':
    main()
