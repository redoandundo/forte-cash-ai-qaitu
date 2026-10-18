"""Create a local runtime environment; optional CUDA Qwen download is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qwen',action='store_true',help='Install NVIDIA CUDA Qwen runtime and download ~4.25 GB base weights')
    args=parser.parse_args()
    env=ROOT/'.venv-run'
    python=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    def run(*arguments):
        subprocess.run([str(python),*arguments],cwd=ROOT,check=True)
    run('-m','pip','install','-r','requirements-runtime.txt')
    run('-m','pip','install','-e','.','--no-deps')
    if args.qwen:
        run('-m','pip','install','torch==2.11.0','torchvision==0.26.0','--index-url','https://download.pytorch.org/whl/cu128')
        run('-m','pip','install','-r','requirements-qwen.txt')
        manifest=json.loads((ROOT/'configs/inference/qwen_verifier_manifest.json').read_text('utf-8'))
        weights=ROOT/manifest['local_dir']/manifest['weights_files'][0]
        valid=False
        if weights.exists():
            with weights.open('rb') as stream:
                valid=hashlib.file_digest(stream,'sha256').hexdigest()==manifest['weights_sha256']
        if not valid:
            run('scripts/ml/download_qwen_verifier.py')
        run('-c','import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print(torch.cuda.get_device_name(0))')
    print('Ready. Set FORTEBANK_API_KEY and run:')
    print(str(python.relative_to(ROOT))+' scripts/run_local.py')


if __name__=='__main__':
    main()
