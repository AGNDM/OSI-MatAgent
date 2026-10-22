"""Explicit release-documented YaRN config and identity checks; run in Slurm."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from comparison.backends import HuggingFaceBackend, _hf_context_extension
from graph_local.backends import _check_shared_identity


class Config:
    def __init__(self, **overrides):
        self.__dict__.update(model_type='qwen2', _name_or_path='Qwen/Qwen2.5-14B-Instruct',
            max_position_embeddings=32768, rope_scaling=None)
        self.__dict__.update(overrides)
    def to_dict(self):
        return deepcopy(self.__dict__)


def test_default_config_object_and_values_are_unchanged():
    original = Config()
    before = original.to_dict()
    result, receipt = _hf_context_extension({}, original)
    assert result is original and result.to_dict() == before and receipt is None


def test_official_yarn_config_is_cloned_and_audited_without_mutating_checkpoint_config():
    original = Config()
    before = original.to_dict()
    result, receipt = _hf_context_extension({'context_extension': 'qwen25-yarn-v1'}, original)
    assert original.to_dict() == before
    assert result is not original
    assert result.max_position_embeddings == 131072
    assert result.rope_scaling == {'factor': 4.0, 'original_max_position_embeddings': 32768,
                                   'type': 'yarn', 'rope_type': 'yarn'}
    assert receipt['operational_max_context'] == 65536
    assert receipt['original_config'] == before
    assert receipt['effective_config'] == result.to_dict()
    assert receipt['original_config_sha256'] != receipt['effective_config_sha256']
    assert receipt['weights_modified'] is receipt['training_performed'] is False


def test_released_qwen_config_passes_installed_transformers_rope_validation():
    """Exercise the real release config and validator without loading weights."""
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import rope_config_validation

    configured = os.environ.get('GRAPH_QWEN25_14B_MODEL')
    if not configured:
        pytest.skip('Set GRAPH_QWEN25_14B_MODEL to the local released Qwen2.5-14B checkpoint; this check reads config only')
    checkpoint = str(Path(configured).expanduser().resolve())
    assert (Path(checkpoint) / 'config.json').is_file(), 'GRAPH_QWEN25_14B_MODEL must contain the released config.json'
    original = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    before = deepcopy(original.to_dict())
    result, receipt = _hf_context_extension(
        {'model_path': checkpoint, 'context_extension': 'qwen25-yarn-v1'}, original)
    rope_config_validation(result)
    assert original.to_dict() == before
    assert result is not original
    assert result.model_type == 'qwen2'
    assert result.max_position_embeddings == 131072
    assert result.rope_scaling['rope_type'] == result.rope_scaling['type'] == 'yarn'
    assert result.rope_scaling['factor'] == 4.0
    assert result.rope_scaling['original_max_position_embeddings'] == 32768
    assert receipt['operational_max_context'] == 65536
    assert receipt['effective_config'] == result.to_dict()


@pytest.mark.parametrize('overrides', [{'model_type': 'qwen3'},
    {'_name_or_path': 'Qwen/Qwen2-7B'}, {'max_position_embeddings': 65536},
    {'rope_scaling': {'type': 'yarn', 'factor': 2}}])
def test_extension_rejects_unsupported_or_already_changed_native_configs(overrides):
    with pytest.raises(ValueError, match='unextended released Qwen2.5'):
        _hf_context_extension({'context_extension': 'qwen25-yarn-v1'}, Config(**overrides))


def test_extension_rejects_operational_context_above_65536():
    with pytest.raises(ValueError, match='cannot exceed65536'):
        _hf_context_extension({'context_extension': 'qwen25-yarn-v1', 'max_context': 131072}, Config())


@pytest.mark.parametrize('declared,loaded', [(None, 'qwen25-yarn-v1'), ('qwen25-yarn-v1', None)])
def test_shared_backend_extension_identity_must_match_exactly(declared, loaded):
    with pytest.raises(ValueError, match='context_extension'):
        _check_shared_identity({'context_extension': declared}, SimpleNamespace(identity={'context_extension': loaded}))
    _check_shared_identity({'context_extension': declared}, SimpleNamespace(identity={'context_extension': declared}))


def test_operational_guard_reserves_output_and_never_silently_truncates():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return 'exact-public-input'
        def __call__(self, prompt, **kwargs):
            assert kwargs['truncation'] is False
            return SimpleNamespace(to=lambda device: SimpleNamespace(input_ids=SimpleNamespace(shape=(1, 64000))))
    backend = object.__new__(HuggingFaceBackend)
    backend.tokenizer = Tokenizer()
    backend.device = 'cpu'
    backend.max_context = 65536
    with pytest.raises(ValueError, match='64000.*2048.*65536'):
        backend._inputs([{'role': 'user', 'content': 'public'}], 2048)
    _, count = backend._inputs([{'role': 'user', 'content': 'public'}], 1024)
    assert count == 64000
