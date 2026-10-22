"""Genuine local decision heads: OpenJev pointer probabilities and APUS depths.

These paths use the released learned decision heads, not ordinary text-token
logits. They make no claim of calibration on instrument tasks. All heavy imports
are lazy, weights must already be staged locally, and evidence is never truncated.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def _allocation():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('Native decision inference requires a Slurm compute allocation')


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def distribution(raw, labels):
    if not isinstance(raw, dict) or set(raw) != set(labels):
        raise ValueError('Native probabilities must cover exactly the offered labels')
    values = {label: float(raw[label]) for label in labels}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError('Invalid native probability')
    mass = sum(values.values())
    if not math.isclose(mass, 1., rel_tol=0, abs_tol=1e-5):
        raise ValueError('Native probability mass differs from one')
    return {label: value / mass for label, value in values.items()}


def _artifact_hashes(root):
    # Authenticate actual model and runtime bytes, including sharded weights.
    names = {'.json', '.safetensors', '.bin', '.model', '.yaml', '.py', '.txt', '.jinja'}
    result = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix not in names or any(part in {'.cache', '__pycache__', '.git'} for part in path.parts):
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        result[path.relative_to(root).as_posix()] = digest.hexdigest()
    return result


class NativeBackend:
    def __init__(self, config):
        self.config = dict(config)
        self.family = config.get('family', 'openjev')
        if self.family not in {'openjev', 'apus'}:
            raise ValueError('Native family must be openjev or apus')
        self.effort = config.get('effort', 'high')
        if self.effort not in {'low', 'high'} or self.family == 'openjev' and self.effort != 'high':
            raise ValueError('Only APUS exposes declared low and high trained-depth paths')
        self.path = Path(config['model_path']).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)
        self.device = config.get('device', 'cuda:0')
        self.dtype = config.get('dtype', 'bfloat16')
        self.runtime = None
        self.identity = {'kind': 'native', 'family': self.family, 'model_path': str(self.path),
            'effort': self.effort, 'device': self.device, 'dtype': self.dtype,
            'mechanism': 'learned_pointer_head' if self.family == 'openjev' else 'trained_depth_candidate_projection',
            'probability_semantics': 'released native distribution; calibration on this benchmark unestablished',
            'context_policy': 'error_without_truncation'}

    def _load(self):
        _allocation()
        if self.runtime is not None:
            return
        import torch
        self.torch = torch
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no implicit CPU fallback')
        started = time.perf_counter()
        self.identity.update(torch=torch.__version__, transformers=_version('transformers'),
                             local_files_sha256=_artifact_hashes(self.path))
        manifest = self.path / 'benchmark-model-manifest.json'
        if not manifest.exists() and self.family == 'apus':
            manifest = self.path.parent / manifest.name
        if manifest.exists():
            self.identity['staged_model_manifest'] = json.loads(manifest.read_text())
        if self.family == 'openjev':
            if _version('wev-ai') != '0.1.0':
                raise RuntimeError('OpenJev native adapter requires wev-ai==0.1.0')
            import wev
            self.runtime = wev.load(str(self.path), device=self.device, dtype=getattr(torch, self.dtype))
            if self.runtime.meta.get('head_type') != 'pointer':
                raise ValueError('Expected the genuine OpenJev pointer head')
            self.identity.update(head_type='pointer', native_package='wev-ai==0.1.0',
                                 native_temperature=self.runtime.meta.get('temperature', 1.0))
        else:
            if _version('transformers') != '5.16.1':
                raise RuntimeError('APUS requires its separate Transformers==5.16.1 environment')
            source = self.path / 'openjet_runtime/__init__.py'
            if not source.is_file():
                raise FileNotFoundError('APUS snapshot lacks its official openjet_runtime')
            namespace = '_instrument_comparison_apus_' + hashlib.sha256(str(source).encode()).hexdigest()[:12]
            spec = importlib.util.spec_from_file_location(namespace, source, submodule_search_locations=[str(source.parent)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = module
            spec.loader.exec_module(module)
            self.runtime = module.OpenJet.from_pretrained(str(self.path), self.device, self.dtype)
            self.identity.update(trained_depths=self.runtime.depths, native_package='snapshot/openjet_runtime')
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)
            self.identity['hardware'] = torch.cuda.get_device_name(torch.device(self.device))
        self.identity['load_wall_time_s'] = time.perf_counter() - started

    def decide(self, messages, option_ids, *, option_descriptions=None, method='native', samples=10,
               seed=0, temperature=.8):
        _allocation()
        started = time.perf_counter()
        labels = list(option_ids)
        metadata = {'method': method, 'native_family': self.family, 'effort': self.effort,
                    'probability_semantics': self.identity['probability_semantics'],
                    'input_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        try:
            if method != 'native':
                raise ValueError('Dedicated decision heads support method=native only')
            if not labels or len(labels) != len(set(labels)) or any(not isinstance(label, str) or not label for label in labels):
                raise ValueError('Options must be distinct nonempty labels')
            if option_descriptions is None or set(option_descriptions) != set(labels):
                raise ValueError('Native decision heads require descriptions for every offered option')
            if self.family == 'apus' and len(labels) > 16:
                raise ValueError('APUS native model supports at most 16 options; menu is not truncated')
            if len(labels) == 1:
                probabilities, input_tokens = {labels[0]: 1.}, 0
                metadata['single_option_deterministic'] = True
            else:
                self._load()
                # Initialization/hash verification is reported separately from
                # each decision's encoding and native-head inference latency.
                started = time.perf_counter()
                state = '\n\n'.join(f"{message['role']}:\n{message['content']}" for message in messages)
                instruction = 'Select the best offered option using the complete public context.'
                if self.family == 'openjev':
                    from wev.api import SystemOneRequest, to_record
                    from wev.model import encode
                    request = SystemOneRequest(state=state, model='alanhuangya/OpenJev-8B',
                        questions={'decision': {'type': 'choice', 'instructions': instruction,
                                              'criteria': {label: option_descriptions[label] for label in labels}}})
                    record, _ = to_record(request)
                    encoded = encode(self.runtime.tokenizer, record, self.runtime.max_state, self.runtime.max_branch, strict=True)
                    with self.torch.inference_mode():
                        raw = self.runtime.model.probs([encoded])[0][0]
                    raw = raw.tolist() if hasattr(raw, 'tolist') else raw
                    probabilities = distribution(dict(zip(labels, raw)), labels)
                    input_tokens = len(encoded['ids'])
                    metadata['native_probabilities_before_display_rounding'] = True
                else:
                    result = self.runtime.decide({'id': 'decision', 'group_id': 'decision', 'primitive': 'choice',
                        'state': state, 'instructions': instruction,
                        'criteria': [{'id': label, 'description': option_descriptions[label]} for label in labels]}, self.effort)
                    probabilities = distribution(result['probabilities'], labels)
                    input_tokens = result['prompt_tokens']
                    metadata.update(executed_layers=result['executed_layers'], projection=result['projection'],
                                    native_calibrated_flag=result['calibrated'])
                if self.device.startswith('cuda'):
                    self.torch.cuda.synchronize(self.device)
            selected = max(probabilities, key=probabilities.get)
            return {'status': 'ok', 'selected_option_id': selected, 'probabilities': probabilities,
                    'confidence': probabilities[selected], 'causes_cited': [], 'input_tokens': input_tokens,
                    'output_tokens': 0, 'latency_s': time.perf_counter() - started,
                    'metadata': {**metadata, 'backend_identity': self.identity}}
        except Exception as error:
            return {'status': 'error', 'selected_option_id': None, 'probabilities': None, 'confidence': None,
                    'causes_cited': [], 'input_tokens': None, 'output_tokens': None,
                    'latency_s': time.perf_counter() - started,
                    'error_type': 'infrastructure_error',
                    'error': f'{type(error).__name__}: {error}',
                    'metadata': {**metadata, 'error_kind': 'infrastructure',
                                 'backend_identity': self.identity}}

    def choose(self, messages, labels, *, request_id=None):
        labels = list(labels)
        result = self.decide(messages, labels, option_descriptions={label: label for label in labels},
                             method=self.config.get('method', 'native'))
        probabilities = result['probabilities']
        return {**result, 'index': labels.index(result['selected_option_id']) if result['status'] == 'ok' else None,
                'probabilities': [probabilities[label] for label in labels] if probabilities else None,
                'usage': {'input_tokens': result['input_tokens'], 'output_tokens': result['output_tokens']},
                'elapsed_s': result['latency_s']}
