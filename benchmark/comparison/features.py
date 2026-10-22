"""Frozen neural features and train-only reference decisions.

Fitted artifacts are local, trusted Python pickles accompanied by a mandatory
SHA256 manifest. They contain only a scaler/classifier and fitting metadata;
neural weights are loaded separately and their identity must match the fit.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import time

from .backends import require_slurm, _sha256, _json_sha256, _validate_ids, _validate_messages

TEXT_PROTOCOL = "complete-public-messages-and-candidate-descriptions-v1"
ARTIFACT_SCHEMA = 2


def _versions():
    return {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy")}


def _source_hashes():
    return {name: _sha256(Path(__file__).with_name(name))
            for name in ("features.py", "feature_models.py", "feature_io.py")}


def _cohort_metadata(train, validation, test):
    groups = [{task.get("group_id", task["task_id"]) for task in subset}
              for subset in (train, validation, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Feature fitting train/validation/test groups overlap")
    return {name: {"task_ids": sorted(task["task_id"] for task in subset),
                   "groups": sorted(group_set)}
            for name, subset, group_set in zip(("train", "validation", "test"),
                                               (train, validation, test), groups)}


def _validate_cohort(cohort):
    if not isinstance(cohort, dict) or set(cohort) != {"train", "validation", "test"}:
        raise ValueError("Feature artifact requires frozen train/validation/test cohort metadata")
    task_sets, group_sets = [], []
    for name in ("train", "validation", "test"):
        subset = cohort[name]
        for field in ("task_ids", "groups"):
            values = subset.get(field) if isinstance(subset, dict) else None
            if (not isinstance(values, list) or not values
                    or any(not isinstance(value, str) or not value for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"Invalid {name} feature cohort {field}")
        task_sets.append(set(subset["task_ids"]))
        group_sets.append(set(subset["groups"]))
    for sets in (task_sets, group_sets):
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("Feature artifact train/validation/test cohorts overlap")


def _validate_feature_manifest(manifest):
    if manifest.get("schema_version") != ARTIFACT_SCHEMA or manifest.get("artifact_type") != "frozen_public_features":
        raise ValueError("Unsupported frozen-feature artifact schema/type")
    if manifest.get("method") not in {"cosine", "linear", "knn"} or manifest.get("text_protocol") != TEXT_PROTOCOL:
        raise ValueError("Feature artifact method/text protocol mismatch")
    if manifest.get("versions") != _versions():
        raise ValueError("Feature artifact NumPy/sklearn/SciPy versions differ from fitting")
    if manifest.get("feature_source_sha256") != _source_hashes():
        raise ValueError("Feature extraction/fitting source differs from fitted artifact")
    _validate_cohort(manifest.get("cohort"))
    train_ids = manifest["cohort"]["train"]["task_ids"]
    validation_ids = manifest["cohort"]["validation"]["task_ids"]
    fit, calibration = manifest.get("fit", {}), manifest.get("calibration", {})
    if fit.get("fitting_scope") != "train_only" or fit.get("task_count") != len(train_ids):
        raise ValueError("Feature fitting scope/count does not match training cohort")
    if fit.get("training_task_id_hash") != _json_sha256(sorted(train_ids)):
        raise ValueError("Feature fitting task identity does not match training cohort")
    if calibration.get("validation_task_id_hash") != _json_sha256(sorted(validation_ids)):
        raise ValueError("Feature calibration task identity does not match validation cohort")
    if manifest.get("fitted_neural_weights") is not False:
        raise ValueError("A frozen feature artifact cannot claim fitted neural weights")


def public_task(messages, ids, descriptions):
    messages = _validate_messages(messages)
    ids = _validate_ids(ids)
    if not isinstance(descriptions, dict) or set(descriptions) != set(ids):
        raise ValueError("Frozen features require every offered candidate description")
    return {"question": json.dumps(messages, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")),
            "options": [{"option_id": key, "text": descriptions[key]} for key in ids]}


class CachedEncoder:
    """Content cache; cold encoding and actual reuse are recorded separately."""

    def __init__(self, encoder):
        self.encoder, self.provenance, self.cache = encoder, dict(encoder.provenance), {}
        self.last_usage = {}

    def encode(self, texts):
        import numpy as np
        texts = list(texts)
        if not texts:
            self.last_usage = {"texts": 0, "cold_texts": 0, "cache_hits": 0, "input_tokens": 0,
                               "cache_policy": "per-process-content-cache-cold-usage-only"}
            return np.empty((0, 0))
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        missing = dict((key, text) for key, text in zip(keys, texts) if key not in self.cache)
        usage = {}
        if missing:
            encoded = self.encoder.encode(list(missing.values()))
            encoded = np.asarray(encoded)
            if encoded.ndim != 2 or len(encoded) != len(missing) or not np.isfinite(encoded).all():
                raise ValueError("Encoder returned an incomplete or non-finite feature matrix")
            usage = dict(self.encoder.last_usage)
            for key, vector in zip(missing, encoded):
                self.cache[key] = vector
        self.last_usage = {**usage, "texts": len(texts), "cold_texts": len(missing),
                           "cache_hits": len(texts) - len(missing),
                           "input_tokens": usage.get("input_tokens", usage.get("content_tokens", 0)),
                           "cache_policy": "per-process-content-cache-cold-usage-only"}
        return np.stack([self.cache[key] for key in keys])


def build_encoder(spec):
    require_slurm()
    from .feature_models import FrozenSentenceTransformerEncoder, FrozenLLMHiddenEncoder
    path = Path(spec["model_path"]).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    kind = spec.get("encoder_kind", "embedding")
    if kind == "embedding":
        encoder = FrozenSentenceTransformerEncoder(path, device=spec.get("device", "cuda:0"))
        files = {str(p.relative_to(path)): _sha256(p) for p in sorted(path.rglob("*"))
                 if p.is_file() and p.suffix in {".safetensors", ".bin", ".json", ".txt", ".model"}}
        identity = {"kind": kind, "model_files_sha256": files,
                    "repository": spec.get("repository"), "revision": spec.get("revision")}
    elif kind == "hf_hidden":
        from .backends import HuggingFaceBackend
        backend = HuggingFaceBackend({**spec, "kind": "hf", "max_new_tokens": 1})
        encoder = FrozenLLMHiddenEncoder(backend.model, backend.tokenizer,
            repository=spec["repository"], revision=spec["revision"],
            device=spec.get("device", "cuda:0"), chunk_tokens=spec.get("chunk_tokens", 512))
        identity = {"kind": kind, "backend": backend.identity}
    else:
        raise ValueError(f"Unknown frozen encoder: {kind}")
    encoder.provenance.update(repository=spec.get("repository"), revision=spec.get("revision"),
                              model_identity=identity, text_protocol=TEXT_PROTOCOL)
    if not encoder.provenance.get("repository") or not encoder.provenance.get("revision"):
        raise ValueError("A frozen encoder requires an explicit repository and revision")
    return CachedEncoder(encoder)


def _encoder_signature(encoder):
    identity = encoder.provenance["model_identity"]
    # Paths and hardware may differ between fitting and evaluation. The model
    # file hashes, feature layer and chunk policy identify the computation.
    if identity["kind"] == "hf_hidden":
        backend = identity["backend"]
        identity = {key: value for key, value in backend.items()
                    if "sha256" in key or key in {"dtype", "model_class", "tokenizer_class"}}
    return _json_sha256({"identity": identity, "text_protocol": TEXT_PROTOCOL,
                         "computation": {key: encoder.provenance.get(key) for key in
                           ("feature_source", "layer", "chunk_policy", "chunk_tokens",
                            "max_chunk_tokens_with_specials", "model_dtype", "torch",
                            "transformers", "sentence_transformers", "tokenizer_class")}})


class FrozenFeatureBackend:
    def __init__(self, spec):
        require_slurm()
        from .feature_models import EmbeddingSimilarity, LinearProbeDecisionModel, KNNDecisionModel
        directory = Path(spec["artifact_path"]).resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        _validate_feature_manifest(manifest)
        artifact = directory / "decision.pkl"
        if _sha256(artifact) != manifest["artifact_sha256"]:
            raise ValueError("Frozen-feature artifact checksum mismatch")
        encoder = build_encoder(spec)
        if _encoder_signature(encoder) != manifest["encoder_signature"]:
            raise ValueError("Feature weights/tokenizer/protocol differ from the fitted artifact")
        with artifact.open("rb") as handle:
            state = pickle.load(handle)
        if (not isinstance(state, dict) or state.get("fit_metadata") != manifest["fit"]
                or state.get("calibration_metadata") != manifest["calibration"]):
            raise ValueError("Feature artifact fitting/calibration state differs from manifest")
        cls = {"cosine": EmbeddingSimilarity, "linear": LinearProbeDecisionModel,
               "knn": KNNDecisionModel}[manifest["method"]]
        self.decision = cls.__new__(cls)
        self.decision.__dict__.update(state)
        self.decision.encoder = encoder
        self.identity = {"backend": "frozen_features", "artifact_path": str(directory),
                         "artifact_sha256": manifest["artifact_sha256"],
                         "manifest_sha256": _sha256(directory / "manifest.json"),
                         "method": manifest["method"], "encoder": encoder.provenance,
                         "fit": manifest["fit"], "calibration": manifest["calibration"],
                         "sklearn": importlib.metadata.version("scikit-learn")}

    def decide(self, messages, option_ids, *, option_descriptions=None, **kwargs):
        require_slurm()
        started = time.perf_counter()
        try:
            task = public_task(messages, option_ids, option_descriptions)
            if len(option_ids) == 1:
                return {"status": "ok", "selected_option_id": option_ids[0],
                        "probabilities": {option_ids[0]: 1.0}, "confidence": 1.0,
                        "input_tokens": 0, "output_tokens": 0, "causes_cited": [],
                        "latency_s": time.perf_counter() - started,
                        "metadata": {"single_option_deterministic": True,
                                     "probability_semantics": "single_offered_option", "text_protocol": TEXT_PROTOCOL}}
            prediction = self.decision.predict(task)
            if prediction.get("status") != "ok":
                raise RuntimeError(prediction.get("error", "Frozen feature prediction failed"))
            probabilities = prediction["probabilities"]
            prediction.update(confidence=probabilities[prediction["selected_option_id"]],
                              latency_s=time.perf_counter() - started)
            # Wall time includes cache reuse and CPU decision work. It is not
            # a measurement of active GPU time.
            prediction["gpu_seconds"] = None
            prediction["metadata"].update(probability_semantics="relative_candidate_softmax",
                                           text_protocol=TEXT_PROTOCOL)
            return prediction
        except Exception as error:
            return {"status": "error", "selected_option_id": None, "probabilities": None,
                    "confidence": None, "input_tokens": None, "output_tokens": None,
                    "latency_s": time.perf_counter() - started,
                    "error": f"{type(error).__name__}: {error}",
                    "metadata": {"error_kind": "infrastructure"}}


class ReferenceBackend:
    def __init__(self, spec):
        require_slurm()
        self.mode = spec.get("reference", "uniform")
        if self.mode not in {"uniform", "majority", "t1"}:
            raise ValueError("Reference must be uniform, majority or t1")
        self.counts, self.fit = {}, {}
        if self.mode == "majority":
            path = Path(spec["artifact_path"]).resolve() / "majority.json"
            manifest = json.loads(path.with_name("majority-manifest.json").read_text())
            if (manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "training_menu_prior"
                    or manifest.get("artifact_sha256") != _sha256(path)):
                raise ValueError("Majority-prior artifact checksum/schema mismatch")
            _validate_cohort(manifest.get("cohort"))
            record = json.loads(path.read_text())
            self.counts, self.fit = record["counts"], record["fit"]
            if (self.fit != manifest.get("fit") or self.fit.get("scope") != "train_only"
                    or self.fit.get("training_task_ids") != manifest["cohort"]["train"]["task_ids"]):
                raise ValueError("Majority-prior fitting scope differs from its training cohort")
            if (not isinstance(self.counts, dict) or not self.counts
                    or any(not isinstance(key, str) or type(value) not in (int, float)
                           or not math.isfinite(value) or value < 0 for key, value in self.counts.items())):
                raise ValueError("Majority-prior counts must be finite and nonnegative")
            self.artifact_sha256 = manifest["artifact_sha256"]
        self.identity = {"backend": "reference", "method": self.mode, "fit": self.fit,
                         "artifact_sha256": getattr(self, "artifact_sha256", None),
                         "semantics": "T1 public-only rules with explicit undefined coverage" if self.mode == "t1" else "menu-label-prior; sequential transfer uses available labels only"}

    def decide(self, messages, option_ids, *, seed=0, **kwargs):
        require_slurm()
        if self.mode == "t1":
            return self._t1(messages, option_ids)
        import random
        ids = _validate_ids(option_ids)
        scores = {key: self.counts.get(key, 0.0) + 1.0 for key in ids}
        total = sum(scores.values())
        probabilities = {key: value / total for key, value in scores.items()}
        selected = max(ids, key=probabilities.get)
        if self.mode == "uniform":
            selected = random.Random(int(_json_sha256(messages)[:16], 16) ^ seed).choice(ids)
        return {"status": "ok", "selected_option_id": selected, "probabilities": probabilities,
                "confidence": probabilities[selected], "causes_cited": [],
                "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0,
                "metadata": {**self.identity,
                             "probability_semantics": "uniform_distribution_over_offered_labels" if self.mode == "uniform" else "laplace_smoothed_training_menu_label_prior"}}

    def _t1(self, messages, option_ids):
        from .t1_rules import predict_t1
        started = time.perf_counter()
        ids = _validate_ids(option_ids)
        try:
            public = json.loads(_validate_messages(messages)[-1]["content"])
            if not isinstance(public, dict) or public.get("input_protocol") != "public-menu-union-v2":
                raise ValueError("T1 rules require the canonical MCQA public input")
            if [option.get("option_id") for option in public.get("options", [])] != ids:
                raise ValueError("T1 public options differ from the offered identifiers")
            result = predict_t1(public)
            probabilities = result.get("probabilities") if result.get("status") == "ok" else None
            result.update(probabilities=probabilities,
                          confidence=probabilities[result["selected_option_id"]] if probabilities else None,
                          input_tokens=0, output_tokens=0, latency_s=time.perf_counter() - started)
            result["metadata"].update(probability_semantics="deterministic_public_rule_reference",
                                      reporting_scope="T1_reference_only", oracle_access=False)
            return result
        except (ValueError, TypeError, KeyError) as error:
            return {"status": "undefined", "selected_option_id": None, "probabilities": None,
                    "confidence": None, "causes_cited": [], "input_tokens": 0, "output_tokens": 0,
                    "latency_s": time.perf_counter() - started,
                    "metadata": {"reason": str(error), "reporting_scope": "T1_reference_only",
                                 "oracle_access": False, "probability_semantics": "none"}}


def fit_artifacts(bundle, output, spec=None, methods=("cosine", "linear", "knn"), label_efficiency=False):
    require_slurm()
    from .policies import load_bundle
    from .run import MCQA_SYSTEM, write_json
    helpers = load_bundle(bundle)
    release_validation = helpers["validate"].validate_bundle(bundle)
    tasks, answers = helpers["common"].load_dataset(Path(bundle) / "mcqa_388/questions.jsonl.gz",
                                                    Path(bundle) / "mcqa_388/answers.jsonl")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    converted = []
    for task in tasks:
        ids = [option["option_id"] for option in task["options"]]
        messages = [{"role": "system", "content": MCQA_SYSTEM},
                    {"role": "user", "content": helpers["prompts"].task_text(task)}]
        item = public_task(messages, ids, {option["option_id"]: json.dumps(
            option, sort_keys=True, ensure_ascii=False, allow_nan=False) for option in task["options"]})
        item.update({key: task[key] for key in ("task_id", "group_id", "split") if key in task})
        converted.append(item)
    train = [task for task in converted if task["split"] == "train"]
    validation = [task for task in converted if task["split"] == "validation"]
    test = [task for task in converted if task["split"].startswith("test")]
    if not train or not validation or not test:
        raise ValueError("Fitting requires disjoint frozen train/validation/test splits")
    cohort = _cohort_metadata(train, validation, test)
    counts = Counter()
    for task in train:
        positive = answers[task["task_id"]]["correct_option_ids"]
        for key in positive:
            counts[key] += 1 / len(positive)
    fit = {"scope": "train_only", "training_task_ids": sorted(task["task_id"] for task in train),
           "training_groups": sorted({task.get("group_id", task["task_id"]) for task in train})}
    write_json(output / "majority.json", {"counts": dict(counts), "fit": fit})
    write_json(output / "majority-manifest.json", {"schema_version": 1,
        "artifact_type": "training_menu_prior", "artifact_sha256": _sha256(output / "majority.json"),
        "fit": fit, "cohort": cohort, "text_protocol": TEXT_PROTOCOL,
        "frozen_release_validation": release_validation})
    if spec is None:
        return
    from .feature_models import EmbeddingSimilarity, LinearProbeDecisionModel, KNNDecisionModel
    encoder = build_encoder(spec)
    constructors = {"cosine": EmbeddingSimilarity, "linear": LinearProbeDecisionModel,
                    "knn": KNNDecisionModel}
    for method in methods:
        decision = constructors[method](encoder)
        decision.fit(train, answers)
        decision.calibrate(validation, answers)
        directory = output / method
        directory.mkdir()
        state = {key: value for key, value in decision.__dict__.items() if key != "encoder"}
        artifact = directory / "decision.pkl"
        with artifact.open("xb") as handle:
            pickle.dump(state, handle, protocol=5)
        write_json(directory / "manifest.json", {
            "schema_version": ARTIFACT_SCHEMA, "artifact_type": "frozen_public_features",
            "text_protocol": TEXT_PROTOCOL, "cohort": cohort, "feature_source_sha256": _source_hashes(),
            "frozen_release_validation": release_validation,
            "method": method, "artifact_sha256": _sha256(artifact),
            "encoder_signature": _encoder_signature(encoder), "encoder_spec": spec,
            "encoder": encoder.provenance, "fit": decision.fit_metadata,
            "calibration": decision.calibration_metadata,
            "training_task_ids": sorted(task["task_id"] for task in train),
            "validation_task_ids": sorted(task["task_id"] for task in validation),
            "fitted_neural_weights": False, "sequential_transfer": "no sequential labels used",
            "versions": _versions()})
        if label_efficiency and method in {"linear", "knn"}:
            from .feature_models import label_efficiency_curve
            curve = label_efficiency_curve(lambda: constructors[method](encoder), train, answers,
                validation, answers, test, answers)
            write_json(directory / "label-efficiency.json", {"results": curve})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--encoder-kind", choices=["embedding", "hf_hidden"], default="embedding")
    parser.add_argument("--repository")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--methods", nargs="+", choices=["cosine", "linear", "knn"],
                        default=["cosine", "linear", "knn"])
    parser.add_argument("--label-efficiency", action="store_true")
    args = parser.parse_args(argv)
    spec = None if not args.model_path else {
        "model_path": args.model_path, "encoder_kind": args.encoder_kind,
        "repository": args.repository, "revision": args.revision, "device": args.device,
        "dtype": args.dtype}
    fit_artifacts(args.bundle, args.output, spec, args.methods, args.label_efficiency)


if __name__ == "__main__":
    main()
