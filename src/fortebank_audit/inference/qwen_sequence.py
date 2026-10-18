"""Local, experimental whole-video Qwen classifier; no legal classification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FRAME_COUNT = 16
MAX_PIXELS = 65536
PROMPT = (
    'These are chronological frames sampled uniformly across one cashier video. '
    'Classify the visible sequence for human review. A = ordinary activity; '
    'B = signs of a possible procedure violation. Cash handling, an ordinary entry, '
    'or another person alone is not enough to establish a violation. Consider the '
    'order of document checks, cash handling, securing cash and equipment, and leaving '
    'the workspace. This is a screening decision, not a legal conclusion. '
    'Reply with exactly one letter, A or B.'
)


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def sample_images(video_path):
    from PIL import Image
    from .video_reader import VideoReader
    reader=VideoReader(video_path)
    meta=reader.metadata()
    end=max(0.,meta.duration_sec-1/max(meta.nominal_fps,1))
    times=[end*i/(FRAME_COUNT-1) for i in range(FRAME_COUNT)]
    frames=list(reader.iter_sample_nearest(times,strict_pts=True))
    if len(frames)!=FRAME_COUNT:
        raise ValueError('Incomplete video sampling')
    return [Image.fromarray(f.image_rgb) for f in frames], [f.timestamp_sec for f in frames]


def encode_images(processor, images):
    messages=[{'role':'user','content':[{'type':'image','image':im} for im in images]+
               [{'type':'text','text':PROMPT}]}]
    batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt')
    if batch.image_grid_thw.shape[0]!=FRAME_COUNT:
        raise ValueError('Unexpected image count')
    return batch


def load_base(model_path):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('Qwen sequence adapter requires CUDA')
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=['visual','lm_head'])
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,
        dtype=torch.bfloat16,device_map={'':'cuda:0'},quantization_config=quant,attn_implementation='sdpa')
    model.config.use_cache=False
    processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True,
        min_pixels=4096,max_pixels=MAX_PIXELS)
    return model,processor


def class_token_ids(processor):
    ids=[processor.tokenizer.encode(letter,add_special_tokens=False) for letter in ('A','B')]
    if any(len(item)!=1 for item in ids) or ids[0]==ids[1]:
        raise ValueError('Class tokens must be distinct single tokens')
    return [item[0] for item in ids]


def class_logits(model,batch,token_ids):
    output=model(**batch.to('cuda'),logits_to_keep=1,use_cache=False)
    return output.logits[:,-1,token_ids].float()


class SequenceClassifier:
    def __init__(self,model_path,adapter_path,*,allow_experimental=False):
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        self.adapter_path=Path(adapter_path)
        self.manifest=json.loads((self.adapter_path/'manifest.json').read_text('utf-8'))
        if not self.manifest.get('quality_gate_passed') and not allow_experimental:
            raise ValueError('Adapter failed release quality gate; use explicit experimental opt-in')
        if checksum(self.adapter_path/'adapter_model.safetensors')!=self.manifest['adapter_sha256']:
            raise ValueError('Adapter checksum mismatch')
        if checksum(Path(model_path)/'model.safetensors')!=self.manifest['base_sha256']:
            raise ValueError('Base checksum mismatch')
        self.model,self.processor=load_base(model_path)
        self.model=prepare_model_for_kbit_training(self.model,use_gradient_checkpointing=False)
        self.model.model.visual.to(dtype=torch.bfloat16)
        self.model=PeftModel.from_pretrained(self.model,self.adapter_path,local_files_only=True).eval()
        self.token_ids=class_token_ids(self.processor)
        self.torch=torch

    def predict(self,video_path):
        import time
        start=time.perf_counter()
        images,timestamps=sample_images(video_path)
        batch=encode_images(self.processor,images)
        with self.torch.inference_mode():
            probability=class_logits(self.model,batch,self.token_ids).softmax(-1)[0,1].item()
        return {'decision':'possible_violation' if probability>=.5 else 'normal_activity',
                'ranking_score':probability,'calibrated_probability':False,
                'processing_sec':time.perf_counter()-start,'sampled_frames':len(timestamps),
                'quality_gate_passed':bool(self.manifest.get('quality_gate_passed'))}


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',required=True)
    parser.add_argument('--model',default='models/pretrained/qwen3-vl-2b-instruct')
    parser.add_argument('--adapter',default='models/qwen-sequence-adapter')
    parser.add_argument('--allow-experimental',action='store_true')
    args=parser.parse_args()
    print(json.dumps(SequenceClassifier(args.model,args.adapter,
        allow_experimental=args.allow_experimental).predict(args.video)))


if __name__=='__main__':
    main()
