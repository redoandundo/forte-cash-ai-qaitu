import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/ml' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contrast_and_duplicate_groups_stay_together():
    module = load('build_qwen_sft')
    config={'splits':{'train':[1,2], 'validation':[3], 'test':[4,5]}}
    assert module.split_for([1,2], config)=='train'
    assert module.split_for([4,5], config)=='test'
    with pytest.raises(ValueError, match='crosses splits'):
        module.split_for([1, 4], config)
    with pytest.raises(ValueError, match='missing'):
        module.split_for([6], config)
    with pytest.raises(ValueError, match='multiple splits'):
        module.split_for([1], {'splits':{'train':[1], 'test':[1]}})


def test_annotation_must_have_actual_nearby_frame():
    module = load('build_qwen_sft')
    target = {'observations':[{'timestamp_sec':10.,'visible_fact':'Example'}]}
    frames = [{'role':'peak','timestamp_sec':10.01}]
    result = module.align_observations(target,[10.],frames)
    assert result['observations'][0]['timestamp_sec'] == 10.01
    assert result['observations'][0]['frame_role'] == 'peak'
    assert target['observations'][0]['timestamp_sec'] == 10.
    with pytest.raises(ValueError,match='no explicitly requested frame'):
        module.align_observations(target,[9.],frames)
    with pytest.raises(ValueError,match='too far'):
        module.align_observations(target,[10.],[{'role':'peak','timestamp_sec':12.}])


def test_optimizer_rejects_holdout_and_unreviewed_sources():
    module = load('train_qwen_adapter')
    record={'split':'train','supervision':'human_timeline_not_frame_verified',
            'frames':[{}],'target':{'decision':'normal_activity'}}
    module.validate_records([record])
    with pytest.raises(ValueError,match='Non-training'):
        module.validate_records([{**record,'split':'test'}])
    with pytest.raises(ValueError,match='provenance'):
        module.validate_records([{**record,'supervision':'automatic_guess'}])
    with pytest.raises(ValueError,match='No training'):
        module.validate_records([])
