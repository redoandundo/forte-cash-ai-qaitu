from types import SimpleNamespace

import pytest

from fortebank_audit.api_runtime import Runtime
from fortebank_audit.inference import benchmark, semantic_pipeline
from fortebank_audit.inference.contracts import PredictorLoadError


def test_selected_bundle_and_release_permission_reach_both_loaders(monkeypatch, tmp_path):
    selected = tmp_path / 'chosen-model'
    monkeypatch.setenv('QAITU_MODEL_BUNDLE', str(selected))
    monkeypatch.setenv('QAITU_ALLOW_DEVELOPMENT_MODEL', 'true')
    monkeypatch.setenv('QAITU_EXPECTED_MODEL_SHA256', 'a' * 64)
    monkeypatch.delenv('QAITU_SEQUENCE_ADAPTER', raising=False)
    calls = []
    def config(bundle, **kwargs):
        calls.append(('config', bundle, kwargs))
        return None, object(), {'sha256': 'a' * 64}
    def predictor(kind, bundle, **kwargs):
        calls.append(('predictor', bundle, kwargs))
        return SimpleNamespace(model_version='selected-model')
    monkeypatch.setattr(benchmark, 'load_pipeline_config', config)
    monkeypatch.setattr(benchmark, 'build_predictor', predictor)
    monkeypatch.setattr(semantic_pipeline, 'build_semantic_verifier_from_env', lambda: None)
    assert Runtime().predictor.model_version == 'selected-model'
    assert [call[1] for call in calls] == [selected.resolve(), selected.resolve()]
    assert all(call[2]['allow_development_model'] for call in calls)


def test_missing_configured_bundle_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv('QAITU_MODEL_BUNDLE', str(tmp_path / 'missing'))
    monkeypatch.setenv('QAITU_ALLOW_DEVELOPMENT_MODEL', 'false')
    with pytest.raises(PredictorLoadError, match='directory does not exist'):
        Runtime()


def test_wrong_checksum_stops_before_model_loading(monkeypatch):
    monkeypatch.setenv('QAITU_EXPECTED_MODEL_SHA256', 'b' * 64)
    monkeypatch.setenv('QAITU_ALLOW_DEVELOPMENT_MODEL', 'false')
    monkeypatch.setattr(benchmark, 'load_pipeline_config', lambda *args, **kwargs: (None, None, {'sha256': 'a' * 64}))
    monkeypatch.setattr(benchmark, 'build_predictor', lambda *args, **kwargs: pytest.fail('Must not load wrong model'))
    with pytest.raises(ValueError, match='checksum'):
        Runtime()


def test_invalid_release_permission_is_not_silently_enabled(monkeypatch):
    monkeypatch.setenv('QAITU_ALLOW_DEVELOPMENT_MODEL', 'sometimes')
    with pytest.raises(ValueError, match='boolean'):
        Runtime()
