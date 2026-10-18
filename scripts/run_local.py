"""Start the API with one worker, explicit local auth and optional Qwen."""
import argparse
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--qwen-semantic',action='store_true')
    parser.add_argument('--qwen-sequence',action='store_true')
    parser.add_argument('--allow-experimental',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('FORTEBANK_API_KEY'):
        parser.error('Set FORTEBANK_API_KEY to a private random key before starting')
    if args.qwen_semantic and args.qwen_sequence:
        parser.error('Choose only one Qwen mode')
    os.chdir(ROOT)
    os.environ['QAITU_VLM_MODE']='qwen' if args.qwen_semantic else 'off'
    os.environ['QAITU_VLM_MODEL_PATH']=str(ROOT/'models/pretrained/qwen3-vl-2b-instruct')
    if args.qwen_sequence:
        os.environ['QAITU_SEQUENCE_ADAPTER']=str(ROOT/'models/qwen-sequence-adapter')
        os.environ['QAITU_ALLOW_EXPERIMENTAL_ADAPTER']='1' if args.allow_experimental else '0'
    else:
        os.environ.pop('QAITU_SEQUENCE_ADAPTER',None)
    # Fail at startup, rather than accept jobs with unavailable model artifacts.
    from fortebank_audit import api_runtime
    api_runtime._runtime=api_runtime.Runtime()
    import uvicorn
    uvicorn.run('fortebank_audit.api:app',host=args.host,port=args.port,workers=1)


if __name__=='__main__':
    main()
