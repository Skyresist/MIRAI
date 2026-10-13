"""Pull Qwen normally, or import checksum-verified registry assets behind a redirect proxy."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE_MODEL, Settings


def checked_download(url: str, destination: Path, digest: str, size: int):
    import requests
    hasher, written = hashlib.sha256(), 0
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with destination.open('wb') as target:
            for chunk in response.iter_content(1024 * 1024):
                target.write(chunk)
                hasher.update(chunk)
                written += len(chunk)
    if written != size or 'sha256:' + hasher.hexdigest() != digest:
        destination.unlink(missing_ok=True)
        raise RuntimeError('Registry asset failed checksum/size verification; refusing to import it.')


def import_registry_model(model: str):
    import requests
    if not re.fullmatch(r'[a-zA-Z0-9_-]+:[a-zA-Z0-9_.-]+', model):
        raise ValueError('Registry import supports only official library model:tag names.')
    name, tag = model.split(':')
    registry = f'https://registry.ollama.ai/v2/library/{name}'
    response = requests.get(f'{registry}/manifests/{tag}', timeout=30)
    response.raise_for_status()
    manifest = response.json()
    with tempfile.TemporaryDirectory(prefix='mirai-model-') as temporary:
        root, assets = Path(temporary), {}
        for layer in manifest['layers']:
            digest = layer['digest']
            if not re.fullmatch(r'sha256:[a-f0-9]{64}', digest):
                raise ValueError('Invalid registry digest')
            kind = layer['mediaType'].rsplit('.', 1)[-1]
            if kind not in {'model', 'template', 'license', 'params'} or kind in assets:
                raise ValueError('Unsupported model layer layout; use native ollama pull instead.')
            path = root / ('weights.gguf' if kind == 'model' else kind)
            print('Downloading and verifying:', kind, flush=True)
            checked_download(f'{registry}/blobs/{digest}', path, digest, layer['size'])
            assets[kind] = path
        if 'model' not in assets:
            raise ValueError('Registry manifest has no model weights')
        lines = [f'FROM {assets["model"]}']
        for kind, directive in [('template', 'TEMPLATE'), ('license', 'LICENSE')]:
            if kind in assets:
                lines.append(directive + ' """' + assets[kind].read_text() + '"""')
        if 'params' in assets:
            for key, value in json.loads(assets['params'].read_text()).items():
                if not re.fullmatch(r'[a-z_]+', key):
                    raise ValueError('Invalid model parameter name')
                for item in value if isinstance(value, list) else [value]:
                    lines.append(f'PARAMETER {key} {json.dumps(item)}')
        modelfile = root / 'Modelfile'
        modelfile.write_text('\n'.join(lines) + '\n')
        subprocess.run(['ollama', 'create', model, '-f', str(modelfile)], check=True)


def pull(model=BASE_MODEL, registry_import=False):
    if registry_import:
        import_registry_model(model)
        return
    import ollama
    client = ollama.Client(host=Settings().ollama_host, timeout=900)
    status = None
    for progress in client.pull(model, stream=True):
        if progress['status'] != status:
            status = progress['status']
            print('Ollama:', status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=BASE_MODEL)
    parser.add_argument('--registry-import', action='store_true', help='Use verified downloads + supported ollama create when the server proxy cannot follow redirects.')
    args = parser.parse_args()
    pull(args.model, args.registry_import)
