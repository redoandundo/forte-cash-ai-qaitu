"""Prepare private timeline-supervised pilot data without sending media externally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLES = ('before', 'start', 'peak', 'end', 'after')

def load_split_config():
    return json.loads((ROOT / 'data/processed/qwen_sft/split_config.json').read_text('utf-8'))


def split_for(ids, config=None):
    config = config or load_split_config()
    mapping = {}
    for name, members in config['splits'].items():
        if name not in {'train','validation','test'}:
            raise ValueError('Unknown split name')
        for identifier in members:
            if identifier in mapping:
                raise ValueError('Source appears in multiple splits')
            mapping[identifier] = name
    if any(i not in mapping for i in ids):
        raise ValueError('Source missing from fixed split')
    splits = {mapping[i] for i in ids}
    if len(splits) != 1:
        raise ValueError('Duplicate/scenario group crosses splits')
    return splits.pop()


def align_observations(target, requested_times, frames):
    result = json.loads(json.dumps(target))
    for obs in result['observations']:
        if obs['timestamp_sec'] not in requested_times:
            raise ValueError('Observation has no explicitly requested frame')
        frame = frames[requested_times.index(obs['timestamp_sec'])]
        if abs(frame['timestamp_sec'] - obs['timestamp_sec']) > .10:
            raise ValueError('Decoded frame too far from annotated action')
        obs['timestamp_sec'] = frame['timestamp_sec']
        obs['frame_role'] = frame['role']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/processed/qwen_sft/pilot_v1')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'data/processed'):
        raise ValueError('Private outputs must stay in ignored data/processed')
    output.mkdir(parents=True, exist_ok=False)
    base = ROOT / 'data/processed/qwen_sft'
    annotations = json.loads((base / 'user_annotations_20261018.json').read_text('utf-8'))
    sources = json.loads((ROOT / 'analysis/dataset_manifest.json').read_text('utf-8'))['sources']
    by_id = {i:s for s in sources for i in s['video_ids']}
    if set(by_id) != {a['video_id'] for a in annotations['records']}:
        raise ValueError('Annotation coverage mismatch')
    pairs = load_split_config()['scenario_groups']
    for pair in pairs:
        split_for(pair)
    audit = []
    for s in sources:
        with (ROOT / s['path']).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != s['sha256']:
            raise ValueError('Source checksum mismatch')
        audit.append({'source_id':s['source_id'], 'sha256':digest, 'video_ids':s['video_ids'],
                      'split':split_for(s['video_ids'])})
    conflicts = []
    for a in annotations['records']:
        s = by_id[a['video_id']]
        if a['label'] != s['label']:
            conflicts.append(a['video_id'])
        if a['event_sec'] is not None and not 0 <= a['event_sec'] < s['duration_sec']:
            raise ValueError('Annotation time outside video')
    if conflicts:
        raise ValueError(f'Label conflicts require review: {conflicts}')
    (output / 'split_manifest.json').write_text(json.dumps({
        'warning':'Development-only split: all sources previously inspected; ONNX training overlap not excluded.',
        'sources':audit, 'scenario_groups':pairs,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    from PIL import Image
    from fortebank_audit.inference.video_reader import VideoReader
    from fortebank_audit.inference.vlm_verifier import (
        EventContext, TemporalFrame, BackendOutput, build_prompt, load_signal_policy, parse_verification_output,
    )
    policy = load_signal_policy(ROOT / 'configs/inference/observable_signals.json')
    pilot = json.loads((base / 'pilot_targets_20261018.json').read_text('utf-8'))
    records = []
    for spec in pilot['records']:
        source = by_id[spec['video_id']]
        folder = output / f"x-{spec['video_id']}"
        folder.mkdir()
        frames = []
        decoded = list(VideoReader(ROOT / source['path']).iter_sample_nearest(spec['sample_times'], strict_pts=True))
        if len(decoded) != 5:
            raise ValueError('Incomplete temporal pack')
        for role, f in zip(ROLES, decoded):
            path = folder / f'{role}.jpg'
            Image.fromarray(f.image_rgb).save(path, quality=95)
            frames.append({'role':role,'timestamp_sec':f.timestamp_sec,'path':path.relative_to(output).as_posix()})
        target = align_observations(spec['target'], spec['sample_times'], frames)
        context = EventContext(frames[0]['timestamp_sec'],frames[-1]['timestamp_sec'],spec['peak_sec'],
                               0.,0,frames[-1]['timestamp_sec']-frames[0]['timestamp_sec'],
                               tuple(TemporalFrame(f['role'],f['timestamp_sec'],output/f['path']) for f in frames))
        parse_verification_output(json.dumps(target,ensure_ascii=False), context=context,
            allowed_signal_ids={s['id'] for s in policy['signals']}, rule_catalog=None,
            model_version='human-timeline-pilot',model_checksum='',backend_output=BackendOutput('',0,0,None),retry_count=0)
        prompt = build_prompt(context,policy)
        prompt = ('Training context sampled from human timeline, not an ONNX prediction. '
                  'ranking_score and supporting_windows are unavailable placeholders, not detector evidence.\n' + prompt)
        records.append({'sample_id':f"x-{spec['video_id']}", 'source_id':source['source_id'],
            'source_sha256':source['sha256'], 'split':split_for(source['video_ids']),
            'supervision':'human_timeline_not_frame_verified', 'frames':frames,'prompt':prompt,'target':target})
    for split in ('train','validation','test'):
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records if r['split']==split),encoding='utf-8')
    summary = {'annotation_rows':len(annotations['records']),'unique_sources':len(sources),
               'label_conflicts':conflicts,'pilot_examples':len(records),
               'pending_alignment_rows':len(annotations['records'])-len(records),
               'not_production_ready':True}
    (output / 'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
