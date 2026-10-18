import hashlib
import json

import pytest

from fortebank_audit.inference.qwen_sequence import SequenceClassifier


def test_failed_training_gate_blocks_adapter_before_gpu_loading(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'quality_gate_passed': False}))
    with pytest.raises(ValueError, match='failed release quality gate'):
        SequenceClassifier(tmp_path / 'missing-base', tmp_path)


def test_experimental_opt_in_still_requires_exact_adapter_checksum(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'quality_gate_passed': False, 'adapter_sha256': hashlib.sha256(b'correct').hexdigest()}))
    (tmp_path / 'adapter_model.safetensors').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='Adapter checksum mismatch'):
        SequenceClassifier(tmp_path / 'missing-base', tmp_path, allow_experimental=True)
