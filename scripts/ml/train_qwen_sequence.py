"""Train whole-video labels without inventing frame-level descriptions."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')


def metrics(rows):
    c=Counter({k:0 for k in ('tp','tn','fp','fn')})
    for r in rows:
        c[{(0,0):'tn',(0,1):'fp',(1,0):'fn',(1,1):'tp'}[(r['label'],r['prediction'])]]+=1
    tp,tn,fp,fn=(c[k] for k in ('tp','tn','fp','fn'))
    f1p=2*tp/max(1,2*tp+fp+fn)
    f1n=2*tn/max(1,2*tn+fp+fn)
    return {**c,'macro_f1':(f1p+f1n)/2,'recall':tp/max(1,tp+fn),'count':len(rows)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='models/checkpoints/qwen-sequence-v1')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--extra-packages',default='tmp/qwen-training-deps')
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--accumulation',type=int,default=1)
    args=parser.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'models/checkpoints'):
        raise ValueError('Checkpoint outside private checkpoint root')
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/args.extra_packages))
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from fortebank_audit.inference.qwen_sequence import (
        sample_images,encode_images,load_base,class_token_ids,class_logits,checksum,FRAME_COUNT,MAX_PIXELS,PROMPT)
    from build_qwen_sft import split_for,load_split_config
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng=random.Random(42)
    start=time.perf_counter()
    sources=json.loads((ROOT/'analysis/dataset_manifest.json').read_text('utf-8'))['sources']
    human=json.loads((ROOT/'data/processed/qwen_sft/user_annotations_20261018.json').read_text('utf-8'))['records']
    labels={r['video_id']:r['label'] for r in human}
    cache=ROOT/'data/processed/qwen_sft/sequence_v1'
    cache.mkdir(parents=True,exist_ok=True)
    records=[]
    pairs=load_split_config()['scenario_groups']
    for group in pairs:
        split_for(group)
    for s in sources:
        source_labels={labels[i] for i in s['video_ids']}
        if len(source_labels)!=1 or next(iter(source_labels))!=s['label']:
            raise ValueError('Conflicting human labels')
        if checksum(ROOT/s['path'])!=s['sha256']:
            raise ValueError('Source checksum mismatch')
        folder=cache/s['source_id']
        folder.mkdir(exist_ok=True)
        images,times=sample_images(ROOT/s['path'])
        files=[]
        for i,im in enumerate(images):
            path=folder/f'{i:02d}.jpg'
            im.save(path,quality=95)
            files.append(path.relative_to(cache).as_posix())
        records.append({'source_id':s['source_id'],'sha256':s['sha256'],'label':next(iter(source_labels)),
            'split':split_for(s['video_ids']),'frames':files,'timestamps':times})
    (cache/'dataset.json').write_text(json.dumps({'records':records,'scenario_groups':pairs,
        'warning':'Weak full-video supervision, not frame annotations. Previously inspected development sources.'},indent=2),encoding='utf-8')
    counts=dict(Counter(r['split'] for r in records))
    print(json.dumps({'phase':'dataset_ready','split_counts':counts,'elapsed_sec':time.perf_counter()-start}),flush=True)
    base='models/pretrained/qwen3-vl-2b-instruct'
    base_sha=checksum(ROOT/base/'model.safetensors')
    reference=json.loads((ROOT/'configs/inference/qwen_verifier_manifest.json').read_text('utf-8'))
    if base_sha!=reference['weights_sha256']:
        raise ValueError('Base checksum mismatch')
    model,processor=load_base(base)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    model.model.visual.to(dtype=torch.bfloat16)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
        target_modules=r'.*language_model.*self_attn\.(q_proj|v_proj)$',bias='none',task_type='CAUSAL_LM'))
    token_ids=class_token_ids(processor)
    encoded=[]
    for r in records:
        images=[]
        for f in r['frames']:
            with Image.open(cache/f) as im:
                images.append(im.convert('RGB'))
        encoded.append(encode_images(processor,images))
    train=[i for i,r in enumerate(records) if r['split']=='train']
    validation=[i for i,r in enumerate(records) if r['split']=='validation']
    test=[i for i,r in enumerate(records) if r['split']=='test']
    if not all((train,validation,test)):
        raise ValueError('Empty split')
    def evaluate(indices,stage):
        model.eval()
        rows=[]
        with torch.no_grad():
            for i in indices:
                begun=time.perf_counter()
                p=class_logits(model,encoded[i],token_ids).softmax(-1)[0,1].item()
                rows.append({'source_id':records[i]['source_id'],'label':records[i]['label'],
                    'prediction':int(p>=.5),'score':p,'sec':time.perf_counter()-begun})
        (out/f'{stage}_private.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        result=metrics(rows)
        print(json.dumps({'phase':stage,**result}),flush=True)
        return result
    baseline=evaluate(validation,'base_validation')
    params=[p for p in model.parameters() if p.requires_grad]
    initial=[p.detach().cpu().clone() for p in params]
    optimizer=torch.optim.AdamW(params,lr=args.learning_rate)
    best=(-1.,-1.)
    steps=0
    optimizer_steps=0
    history=[]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        rng.shuffle(train)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index,i in enumerate(train):
            logits=class_logits(model,encoded[i],token_ids)
            loss=F.cross_entropy(logits,torch.tensor([records[i]['label']],device='cuda'))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            group_start=(batch_index//args.accumulation)*args.accumulation
            group_size=min(args.accumulation,len(train)-group_start)
            (loss/group_size).backward()
            norm=0.
            if (batch_index+1)%args.accumulation==0 or batch_index+1==len(train):
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps+=1
                optimizer.zero_grad(set_to_none=True)
            steps+=1
            row={'step':steps,'epoch':epoch+1,'loss':loss.item(),'grad_norm':float(norm),
                 'elapsed_sec':round(time.perf_counter()-start,2)}
            with (out/'train_log.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
            if steps%5==0:
                print(json.dumps(row),flush=True)
        result=evaluate(validation,f'epoch_{epoch+1}_validation')
        history.append(result)
        rank=(result['macro_f1'],result['recall'])
        if rank>best:
            best=rank
            best_epoch=epoch+1
            model.save_pretrained(out/'best',safe_serialization=True)
    delta=max(float((p.detach().cpu()-a).abs().max()) for p,a in zip(params,initial))
    if delta<=0:
        raise RuntimeError('Adapter unchanged')
    # Load best adapter in-place; selection used validation only.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model,load_file(out/'best/adapter_model.safetensors'))
    selected=evaluate(validation,'selected_validation')
    with model.disable_adapter():
        base_test=evaluate(test,'base_test')
    final=evaluate(test,'selected_test')
    gate=(selected['macro_f1']>baseline['macro_f1'] and selected['recall']>=baseline['recall']
          and final['macro_f1']>=.8 and final['recall']>=.9 and final['macro_f1']>base_test['macro_f1'])
    summary={'task':'whole-video binary screening; not event localization or semantic JSON generation',
        'quality_gate_passed':gate,'base_sha256':base_sha,
        'adapter_sha256':checksum(out/'best/adapter_model.safetensors'),
        'base_model_id':reference['model_id'],'base_revision':reference['revision'],
        'split_counts':counts,'epochs':args.epochs,'selected_epoch':best_epoch,
        'optimizer_steps':optimizer_steps,'training_examples_processed':steps,
        'frame_count':FRAME_COUNT,'max_pixels':MAX_PIXELS,'prompt':PROMPT,'token_ids':token_ids,
        'base_validation':baseline,'selected_validation':selected,'base_test':base_test,'selected_test':final,
        'max_weight_change':delta,'peak_allocated_vram_gib':torch.cuda.max_memory_allocated()/1024**3,
        'total_elapsed_sec':time.perf_counter()-start,'seed':42,'learning_rate':args.learning_rate,
        'gradient_accumulation':args.accumulation,
        'evaluation_warning':'Small development holdout, not untouched external evaluation. No per-frame grounding learned.'}
    (out/'best/manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
