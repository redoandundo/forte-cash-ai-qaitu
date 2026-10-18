"""One-click host for an already installed local ONNX + Qwen server."""
import os
import argparse
from pathlib import Path
import secrets
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def choose_port(start=8000):
    for port in range(start, start + 20):
        with socket.socket() as probe:
            try:
                probe.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    raise RuntimeError('No free site port in range 8000-8019')


def access_key(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'site.key'
    if not path.exists():
        with path.open('x', encoding='utf-8') as stream:
            stream.write(secrets.token_urlsafe(32))
    value = path.read_text('utf-8').strip()
    if not value:
        raise RuntimeError('Local site key is empty')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-only', action='store_true')
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not (ROOT / 'models/pretrained/qwen3-vl-2b-instruct/model.safetensors').exists():
        raise RuntimeError('Qwen is missing on this server. Run scripts/setup_local.py --qwen once.')
    key = os.environ.get('FORTEBANK_API_KEY') or access_key(ROOT / 'secrets')
    os.environ['FORTEBANK_API_KEY'] = key
    os.environ['QAITU_VLM_MODE'] = 'qwen'
    os.environ['QAITU_VLM_MODEL_PATH'] = str(ROOT / 'models/pretrained/qwen3-vl-2b-instruct')
    os.environ.pop('QAITU_SEQUENCE_ADAPTER', None)
    if args.backend_only:
        os.environ['QAITU_BACKEND_ONLY'] = '1'
    port = args.port or choose_port()
    local = f'http://127.0.0.1:{port}/#key={key}'
    addresses = sorted({item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
                        if not item[4][0].startswith('127.')})
    links = [local] + [f'http://{address}:{port}/#key={key}' for address in addresses]
    private = ROOT / 'secrets'
    private.mkdir(exist_ok=True)
    if not args.backend_only:
        (private / 'tester-access.txt').write_text(
            'Local browser (first link); same-network tester (other links).\n'
            'These links grant access. Share only with your tester.\n' + '\n'.join(links), encoding='utf-8')
    print('Loading ONNX + Qwen already on this server; no model download.', flush=True)
    from fortebank_audit import api_runtime
    api_runtime._runtime = api_runtime.Runtime()
    def open_when_ready():
        for _ in range(60):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                    if response.status == 200:
                        webbrowser.open(local)
                        return
            except OSError:
                time.sleep(1)
    if not args.backend_only:
        threading.Thread(target=open_when_ready, daemon=True).start()
    print(f'Server port: {port}. Backend-only: {args.backend_only}', flush=True)
    print('Keep this process running. The tester only needs a browser.', flush=True)
    import uvicorn
    uvicorn.run('fortebank_audit.api:app', host='127.0.0.1' if args.backend_only else '0.0.0.0',
                port=port, workers=1)


if __name__ == '__main__':
    main()
