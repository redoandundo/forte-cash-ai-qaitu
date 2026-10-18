"""Local QLoRA pilot. Private data/checkpoints only; does not alter inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_records(records):
    if not records:
        raise ValueError('No training examples')
    for r in records:
        if r['split'] != 'train':
            raise ValueError('Non-training source passed to optimizer')
        if r.get('supervision') != 'human_timeline_not_frame_verified':
            raise ValueError('Unsupported supervision provenance')
        if not r.get('frames') or not r.get('target'):
            raise ValueError('Missing image input or answer')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default='data/processed/qwen_sft/pilot_v1/train.jsonl')
    parser.add_argument('--model',default='models/pretrained/qwen3-vl-2b-instruct')
    parser.add_argument('--output',default='models/checkpoints/qwen-qlora-pilot-v1')
    parser.add_argument('--extra-packages',default='tmp/qwen-training-deps')
    parser.add_argument('--steps',type=int,default=6)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError('steps must be positive')
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'models/checkpoints'):
        raise ValueError('Adapter must stay under ignored models/checkpoints')
    if not args.verify_only:
        output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT / args.extra_packages))
    import torch
    import torch.nn.functional as F
    import peft
    import bitsandbytes
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from PIL import Image

    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this pilot')
    data_path = ROOT / args.data
    records = [json.loads(line) for line in data_path.read_text('utf-8').splitlines() if line]
    validate_records(records)
    manifest = json.loads((ROOT / 'configs/inference/qwen_verifier_manifest.json').read_text('utf-8'))
    base_sha = sha(ROOT / args.model / 'model.safetensors')
    if base_sha != manifest['weights_sha256']:
        raise ValueError('Base weights checksum mismatch')
    config = dict(base_model=args.model,base_sha256=base_sha,data=args.data,data_sha256=sha(data_path),
                  seed=42,steps=args.steps,learning_rate=1e-4,lora_r=8,lora_alpha=16,
                  min_pixels=4096,max_pixels=65536,batch_size=1,quantization='nf4-double-quant',
                  torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
                  bitsandbytes=bitsandbytes.__version__,cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0),production_ready=False,
                  warning='Three timeline-supervised examples; no held-out quality claim, no insufficient_evidence class yet')
    print(json.dumps({'phase':'loading','gpu':config['gpu'],'examples':len(records)}),flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=['visual','lm_head'])
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model,local_files_only=True,
        quantization_config=quant,device_map={'':'cuda:0'},dtype=torch.bfloat16,attn_implementation='sdpa')
    processor = AutoProcessor.from_pretrained(args.model,local_files_only=True,
        min_pixels=config['min_pixels'],max_pixels=config['max_pixels'])
    if args.verify_only:
        model = PeftModel.from_pretrained(model,output,local_files_only=True)
    else:
        model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
        # Keep the frozen vision tower in bf16; PEFT otherwise casts it to fp32.
        model.model.visual.to(dtype=torch.bfloat16)
        model = get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,
            target_modules=r'.*language_model.*self_attn\.(q_proj|v_proj)$',
            bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if not args.verify_only and (not trainable or any('lora_' not in n or 'visual' in n for n in trainable)):
        raise RuntimeError('Unexpected trainable parameter selection')
    print(json.dumps({'phase':'loaded','trainable_parameters':sum(p.numel() for p in trainable.values())}),flush=True)

    def encode(record):
        images=[]
        for frame in record['frames']:
            with Image.open(data_path.parent / frame['path']) as im:
                images.append(im.convert('RGB'))
        messages=[{'role':'user','content':[{'type':'image','image':im} for im in images]+
                  [{'type':'text','text':record['prompt']}]}]
        prefix=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt')
        messages.append({'role':'assistant','content':[{'type':'text','text':json.dumps(record['target'],ensure_ascii=False,separators=(',',':'))}]})
        batch=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,
            return_dict=True,return_tensors='pt')
        prefix_len=prefix.input_ids.shape[1]
        if not torch.equal(prefix.input_ids,batch.input_ids[:,:prefix_len]):
            raise ValueError('Chat-template prefix mismatch; cannot mask prompt safely')
        if 'pixel_values' not in batch or batch.image_grid_thw.shape[0] != len(images):
            raise ValueError('Visual input missing')
        return batch,prefix_len

    encoded=[encode(r) for r in records]
    print(json.dumps({'phase':'encoded','tokens':[b.input_ids.shape[1] for b,p in encoded],
                     'answer_tokens':[b.input_ids.shape[1]-p for b,p in encoded]}),flush=True)
    def loss_for(item):
        batch,prefix=item
        batch=batch.to('cuda')
        n=batch.input_ids.shape[1]-prefix
        # Only materialize answer logits; prompt and image tokens have no loss.
        out=model(**batch,logits_to_keep=n+1,use_cache=False)
        logits=out.logits[:,:-1,:].float()
        labels=batch.input_ids[:,prefix:]
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))
        return loss

    if args.verify_only:
        model.eval()
        with torch.no_grad():
            loss=float(loss_for(encoded[0]))
        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError('Reloaded adapter has non-finite loss')
        result={'reload_success':True,'first_train_example_loss':loss,'adapter_sha256':sha(output/'adapter_model.safetensors')}
        (output/'reload_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
        return

    (output/'training_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
    optimizer=torch.optim.AdamW(trainable.values(),lr=config['learning_rate'])
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    history=[]
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss=loss_for(encoded[step%len(encoded)])
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable.values()):
            raise RuntimeError('Non-finite gradients')
        norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.)
        if norm <= 0:
            raise RuntimeError('Zero gradient: training is disconnected')
        optimizer.step()
        row={'step':step+1,'sample_id':records[step%len(records)]['sample_id'],
             'loss':float(loss.detach()),'grad_norm':float(norm),
             'elapsed_sec':round(time.perf_counter()-started,2),
             'peak_vram_gib':round(torch.cuda.max_memory_allocated()/1024**3,3)}
        history.append(row)
        with (output/'train_log.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    max_delta=max(float((p.detach().cpu()-initial[n]).abs().max()) for n,p in trainable.items())
    if max_delta <= 0:
        raise RuntimeError('Adapter weights did not change')
    model.save_pretrained(output,safe_serialization=True)
    processor.save_pretrained(output/'processor')
    model.eval()
    with torch.no_grad():
        final_loss=float(loss_for(encoded[0]))
    summary={'completed_optimizer_steps':len(history),'max_adapter_weight_change':max_delta,
             'first_example_loss_before':history[0]['loss'],'first_example_loss_after':final_loss,
             'adapter_sha256':sha(output/'adapter_model.safetensors'),
             'adapter_bytes':(output/'adapter_model.safetensors').stat().st_size,
             'peak_vram_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),
             'elapsed_sec':round(time.perf_counter()-started,2),'production_ready':False,
             'quality_evaluation':'Not performed. Training loss is not held-out accuracy.'}
    (output/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
