"""Optional Qwen label-logit policy using the frozen bundled implementation.

Set SEQUENTIAL_MODEL_PATH to a local copy of Qwen2.5-3B-Instruct revision
aa8e72537993ba99e69dfaafa59ed015b17504d1. Model weights and GPU dependencies are
not distributed here. The model loader is offline. Run only in a GPU Slurm job
on SCITAS, using the project's pinned model environment.

    python eval/sequential.py run --policy examples.sequential_qwen:make_policy \
        --mode both --output output/qwen-sequential

SEQUENTIAL_DEVICE defaults to cuda:0; SEQUENTIAL_MAX_CONTEXT defaults to 30000;
SEQUENTIAL_MENU_SEED defaults to 42. The full policy identity and actual local
model-file checksums are recorded in each completed episode.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_SHARED = {}


def _model_hashes(directory):
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model"}:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result[path.name] = digest.hexdigest()
    return result


def make_policy(context):
    from instrument_bench.local_llm import LocalLLM
    from stepwise_choice.policies import LocalChoicePolicy

    model_path = Path(os.environ["SEQUENTIAL_MODEL_PATH"]).resolve()
    device = os.environ.get("SEQUENTIAL_DEVICE", "cuda:0")
    max_context = int(os.environ.get("SEQUENTIAL_MAX_CONTEXT", "30000"))
    seed = int(os.environ.get("SEQUENTIAL_MENU_SEED", "42"))
    key = (str(model_path), device, max_context)
    if key not in _SHARED:
        llm = LocalLLM(str(model_path), device=device, max_context=max_context)
        llm.identity["local_model_files_sha256"] = _model_hashes(model_path)
        _SHARED[key] = llm
    return LocalChoicePolicy(context["menu_builder"], llm=_SHARED[key],
                             max_context=max_context, seed=seed)


factory = make_policy
