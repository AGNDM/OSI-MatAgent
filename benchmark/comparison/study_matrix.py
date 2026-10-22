"""Predeclare the SCITAS local study; generation is compute-node-only."""
from __future__ import annotations

import argparse
from pathlib import Path

from .run import require_allocation, write_json, validate_config


def matrix(project, bundle, feature_root=None, qwen3=None):
    project = Path(project).resolve()
    experiments = []
    qwen = {"kind": "hf", "model_path": str(project / "external/models/qwen2.5-3b"),
            "device": "cuda:0", "dtype": "float16", "max_context": 32768,
            "max_new_tokens": 192}

    def add(name, method, backend, *, seed=0, architecture="direct", **extra):
        family = name.rsplit("-seed", 1)[0] if "-seed" in name else name
        experiments.append({"name": name, "analysis_family": family, "method": method,
            "architecture": architecture, "backend": backend, "seed": seed,
            "samples": 10, "temperature": .8, "menu_seed": 42, **extra})

    for model_name, backend in [("qwen25-3b-hf", qwen)] + ([] if qwen3 is None else [
            ("qwen3-4b-hf", {**qwen, "model_path": str(Path(qwen3).resolve())})]):
        for method in ("choice", "logprobs", "verbalized"):
            add(f"{model_name}-{method}-seed0", method, backend)
        add(f"{model_name}-verbalized-cited-seed0", "verbalized", {**backend, "cite_causes": True})
        for seed in (0, 1, 2):
            add(f"{model_name}-sampling-seed{seed}", "sampling", backend, seed=seed)

    ollama = {"kind": "ollama", "model": "qwen25-3b-local:imported",
              "tokenizer_path": qwen["model_path"], "max_context": 32768,
              "max_new_tokens": 192, "timeout_s": 900}
    for method in ("choice", "verbalized"):
        add(f"qwen25-3b-ollama-{method}-seed0", method, ollama)
    add("qwen25-3b-ollama-verbalized-cited-seed0", "verbalized", {**ollama, "cite_causes": True})
    for seed in (0, 1, 2):
        add(f"qwen25-3b-ollama-sampling-seed{seed}", "sampling", ollama, seed=seed)

    openjev = {"kind": "native", "family": "openjev",
               "model_path": str(project / "external/models/openjev8b"),
               "device": "cuda:0", "max_context": 4096}
    apus = {"kind": "native", "family": "apus",
            "model_path": str(project / "external/models/apus-reflex/4B-5949"),
            "device": "cuda:0"}
    add("openjev8b-native-seed0", "native", openjev)
    for effort in ("low", "high"):
        add(f"apus4b-native-{effort}-seed0", "native", {**apus, "effort": effort})

    expert = {**qwen, "max_new_tokens": 384}
    for name, backend, method in (
            ("qwen", expert, "logprobs"), ("openjev", openjev, "native"),
            ("apus-low", {**apus, "effort": "low"}, "native"),
            ("apus-high", {**apus, "effort": "high"}, "native")):
        add(f"agentic-qwen25-to-{name}-seed0", method, backend,
            architecture="expert_decision", expert=expert, max_candidates=4,
            expert_max_new_tokens=384)

    add("reference-uniform-seed0", "native", {"kind": "reference", "reference": "uniform"},
        reporting_role="reference")
    if feature_root:
        root = Path(feature_root).resolve()
        embedding = {"kind": "features", "encoder_kind": "embedding",
            "model_path": str(project / "external/models/minilm-l6"),
            "repository": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "device": "cuda:0"}
        for method in ("cosine", "linear", "knn"):
            add(f"minilm-{method}-seed0", "native",
                {**embedding, "artifact_path": str(root / "minilm" / method)},
                fitted_scope="mcqa-train232-validation42; sequential transfer without fitting")
        add("qwen25-hidden-linear-seed0", "native", {
            **qwen, "kind": "features", "encoder_kind": "hf_hidden",
            "repository": "Qwen/Qwen2.5-3B-Instruct",
            "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "artifact_path": str(root / "qwen25-hidden/linear")},
            fitted_scope="mcqa-train232-validation42; sequential transfer without fitting")
        add("reference-majority-seed0", "native", {"kind": "reference", "reference": "majority",
            "artifact_path": str(root / "minilm")}, reporting_role="reference")
    add("reference-t1-rules", "native", {"kind": "reference", "reference": "t1"},
        tracks=["mcqa"], reporting_role="reference", reporting_scope="T1_reference")
    return {"schema_version": 1, "bundle": str(Path(bundle).resolve()),
        "study_protocol": "local-instrument-study-20261022-v1",
        "experiments": experiments,
        "tracks": {"mcqa": {"split": "all"}, "sequential": {"mode": "both"}},
        "analysis": {"primary_mcqa": "heldout114", "all388": "includes training and validation diagnostics",
            "sequential": "all120 authored cases; no frozen train/test split",
            "groups": "authored instrument failure mechanism, condition and workflow",
            "sampling_seeds": [0, 1, 2], "menu_seed": 42,
            "expert_review_completed": False,
            "hosted_inference_calls": 0}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--qwen3", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_allocation()
    if args.output.exists():
        raise ValueError("Matrix output must be fresh")
    config = matrix(args.project, args.bundle, args.feature_root, args.qwen3)
    validate_config(config)
    write_json(args.output, config)


if __name__ == "__main__":
    main()
