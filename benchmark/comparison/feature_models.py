"""Decision mechanisms fitted only on frozen public-evidence features.

Neural encoders process every input token using explicit chunk pooling. A hashed
or bag-of-words encoder is not silently substituted when model loading fails.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

from .feature_io import option_text, solver_view, task_text

PROJECT = Path(__file__).resolve().parents[1]
MINILM_REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm < 1e-15:
        raise ValueError("Encoder returned an invalid or zero feature vector")
    return vector / norm


def _softmax(scores, temperature=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Expected finite scores for at least two candidates")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    values = scores / temperature
    values = np.exp(values - values.max())
    return values / values.sum()


def _split_guard(tasks, role):
    expected = "train" if role == "train" else "validation"
    for task in tasks:
        if "split" in task and task["split"] != expected:
            raise ValueError(f"{role} received a task from split {task['split']!r}")


def _identity(task):
    return task.get("task_id", hashlib.sha256(task_text(task).encode()).hexdigest())


def _group(task):
    return task.get("group_id", _identity(task))


def _oracle(oracles, task):
    record = oracles[task["task_id"]]
    result = set(record["correct_option_ids"])
    available = {option["option_id"] for option in task["options"]}
    if not result or not result <= available:
        raise ValueError("Training/calibration labels must reference offered options")
    return result


def _signature(tasks):
    ids = sorted(_identity(task) for task in tasks)
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


class FrozenSentenceTransformerEncoder:
    """Pinned MiniLM sentence features with explicit complete-input pooling.

    Chunk token IDs are passed directly to the SentenceTransformer forward path;
    decoding/re-tokenization cannot discard a boundary wordpiece. Each chunk is
    normalized, then pooled with content-token-count weights and normalized.
    """

    def __init__(self, model_path=None, *, device="cpu", batch_size=16):
        import torch
        from sentence_transformers import SentenceTransformer

        path = Path(model_path or PROJECT / "external/models/minilm-l6").resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Missing frozen embedding model: {path}")
        self.model = SentenceTransformer(str(path), device=device, local_files_only=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.torch, self.device, self.batch_size = torch, device, int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.tokenizer = self.model.tokenizer
        self.max_length = int(self.model.max_seq_length)
        self.chunk_size = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        if self.chunk_size < 1:
            raise ValueError("Invalid embedding context size")
        self.last_usage = {}
        self.provenance = {
            "feature_source": "frozen_sentence_transformer", "neural": True,
            "repository": MINILM_REPOSITORY,
            "revision": MINILM_REVISION if path == (PROJECT / "external/models/minilm-l6").resolve() else None,
            "local_path": str(path), "device": device,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "model_dtype": str(next(self.model.parameters()).dtype),
            "tokenizer_class": type(self.tokenizer).__module__ + "." + type(self.tokenizer).__qualname__,
            "max_chunk_tokens_with_specials": self.max_length,
            "chunk_policy": "all_tokens_nonoverlapping_content_weighted_mean_of_normalized_chunks",
            "hidden_weight_training": False,
        }

    def encode(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))
        chunks, owners, weights = [], [], []
        token_count = 0
        for index, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
            token_count += len(ids)
            slices = [ids[start:start + self.chunk_size]
                      for start in range(0, len(ids), self.chunk_size)] or [[]]
            for part in slices:
                chunks.append(self.tokenizer.prepare_for_model(
                    part, add_special_tokens=True, truncation=False,
                    return_attention_mask=True, return_token_type_ids=True))
                owners.append(index)
                weights.append(max(1, len(part)))
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(chunks), self.batch_size):
                padded = self.tokenizer.pad(chunks[start:start + self.batch_size],
                                             padding=True, return_tensors="pt")
                if padded["input_ids"].shape[1] > self.max_length:
                    raise AssertionError("Embedding chunk exceeded the recorded context policy")
                features = {key: value.to(self.device) for key, value in padded.items()}
                encoded = self.model(features)["sentence_embedding"]
                encoded = self.torch.nn.functional.normalize(encoded.float(), dim=-1)
                vectors.extend(encoded.cpu().numpy())
        output = []
        owners = np.asarray(owners)
        weights = np.asarray(weights, dtype=float)
        vectors = np.asarray(vectors)
        for index in range(len(texts)):
            mask = owners == index
            output.append(_unit(np.average(vectors[mask], axis=0, weights=weights[mask])))
        self.last_usage = {"texts": len(texts), "content_tokens": token_count,
                           "input_tokens": token_count + len(chunks) * self.tokenizer.num_special_tokens_to_add(pair=False),
                           "chunks": len(chunks), "truncated_tokens": 0}
        return np.stack(output)


class FrozenLLMHiddenEncoder:
    """Frozen local-LLM hidden-state features from an already loaded model.

    Reuses an evaluator's local model/tokenizer without fitting its parameters.
    Every token contributes through non-overlapping mean-pooled chunks. This is
    a separate feature source from SentenceTransformer embeddings.
    """

    def __init__(self, model, tokenizer, *, repository, revision, device="cuda:0",
                 chunk_tokens=512, layer=-1):
        import torch

        if not repository or not revision:
            raise ValueError("Frozen hidden features require explicit model provenance")
        self.model, self.tokenizer, self.torch = model, tokenizer, torch
        self.device, self.chunk_tokens, self.layer = device, int(chunk_tokens), int(layer)
        if self.chunk_tokens < 1:
            raise ValueError("chunk_tokens must be positive")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.last_usage = {}
        self.provenance = {
            "feature_source": "frozen_local_llm_hidden_state", "neural": True,
            "repository": repository, "revision": revision, "layer": layer,
            "device": device, "chunk_tokens": self.chunk_tokens,
            "chunk_policy": "all_tokens_nonoverlapping_content_weighted_hidden_mean",
            "hidden_weight_training": False,
            "model_dtype": str(next(self.model.parameters()).dtype),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "tokenizer_class": type(self.tokenizer).__module__ + "." + type(self.tokenizer).__qualname__,
        }

    def encode(self, texts):
        outputs, content_tokens, chunk_count = [], 0, 0
        with self.torch.inference_mode():
            for text in texts:
                ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
                if not ids:
                    raise ValueError("Cannot extract hidden features from an empty encoding")
                total, weight = None, 0
                content_tokens += len(ids)
                for start in range(0, len(ids), self.chunk_tokens):
                    part = ids[start:start + self.chunk_tokens]
                    input_ids = self.torch.tensor([part], dtype=self.torch.long, device=self.device)
                    result = self.model(input_ids=input_ids, attention_mask=self.torch.ones_like(input_ids),
                                        output_hidden_states=True, return_dict=True, use_cache=False)
                    hidden = result.hidden_states[self.layer][0].float().mean(dim=0).cpu().numpy()
                    total = hidden * len(part) if total is None else total + hidden * len(part)
                    weight += len(part)
                    chunk_count += 1
                outputs.append(_unit(total / weight))
        self.last_usage = {"texts": len(outputs), "content_tokens": content_tokens,
                           "chunks": chunk_count, "truncated_tokens": 0}
        return np.stack(outputs)


class PrecomputedFeatureEncoder:
    """Strict content-keyed frozen vectors; missing evidence never falls back."""

    def __init__(self, vectors, *, provenance):
        if not provenance.get("feature_source") or not provenance.get("revision"):
            raise ValueError("Precomputed features require source and revision provenance")
        self.vectors, self.provenance = vectors, dict(provenance)
        self.last_usage = {}

    def encode(self, texts):
        result = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest not in self.vectors:
                raise KeyError(f"No precomputed feature for public-content hash {digest}")
            result.append(_unit(self.vectors[digest]))
        self.last_usage = {"texts": len(result), "precomputed": True}
        return np.stack(result)


class _FeatureDecision:
    name = "feature_decision"

    def __init__(self, encoder=None):
        self.encoder = encoder or FrozenSentenceTransformerEncoder()
        provenance = getattr(self.encoder, "provenance", None)
        if provenance is None:
            provenance = getattr(self.encoder, "identity", None)
        if not isinstance(provenance, dict):
            raise ValueError("A frozen encoder requires provenance or identity metadata")
        self.feature_identity = dict(provenance)
        self.training_groups, self.training_task_ids = set(), set()
        self.temperature = 1.0
        self.fit_metadata = {"trained": False}
        self.calibration_metadata = {"fitted": False}

    def _features(self, task):
        view = solver_view(task)
        options = view.get("options", [])
        ids = [option["option_id"] for option in options]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("Decision options must have distinct IDs and at least two members")
        # Private task/group/split metadata are excluded by solver_view. Public
        # menu labels and the complete public message context remain available;
        # each public candidate description is additionally encoded separately.
        texts = [task_text(view, include_options=False)] + [option_text(option) for option in options]
        embeddings = np.asarray(self.encoder.encode(texts), dtype=float)
        if embeddings.ndim != 2 or len(embeddings) != len(texts):
            raise ValueError("Encoder returned an invalid feature matrix")
        embeddings = np.stack([_unit(vector) for vector in embeddings])
        state, candidates = embeddings[0], embeddings[1:]
        cosine = candidates @ state
        repeated = np.repeat(state[None], len(candidates), axis=0)
        features = np.concatenate([repeated, candidates, np.abs(repeated - candidates),
                                   repeated * candidates, cosine[:, None]], axis=1)
        return ids, features, cosine

    def _record_training(self, tasks):
        _split_guard(tasks, "train")
        self.training_groups = {_group(task) for task in tasks}
        self.training_task_ids = {_identity(task) for task in tasks}
        self.fit_metadata = {"trained": True, "task_count": len(tasks),
                             "scenario_groups": len(self.training_groups),
                             "training_task_id_hash": _signature(tasks),
                             "feature_source": dict(self.feature_identity),
                             "fitting_scope": "train_only"}

    def _check_calibration(self, tasks):
        _split_guard(tasks, "calibration")
        if self.training_groups & {_group(task) for task in tasks}:
            raise ValueError("Calibration and training scenario groups overlap")
        if self.training_task_ids & {_identity(task) for task in tasks}:
            raise ValueError("Calibration and training task IDs overlap")

    def fit(self, train_tasks, oracles):
        train_tasks = list(train_tasks)
        self._record_training(train_tasks)
        return self

    def calibrate(self, val_tasks, oracles):
        val_tasks = list(val_tasks)
        self._check_calibration(val_tasks)
        self.calibration_metadata = {"fitted": False, "reason": "No post-fit calibration parameter for this method",
                                     "validation_tasks": len(val_tasks), "validation_task_id_hash": _signature(val_tasks)}
        return self

    def _scores(self, task):
        raise NotImplementedError

    def predict(self, public_task):
        started = time.perf_counter()
        try:
            ids, scores = self._scores(public_task)
            probabilities = _softmax(scores, self.temperature)
            selected = ids[int(np.argmax(probabilities))]
            elapsed = time.perf_counter() - started
            device = self.feature_identity.get("device", "cpu")
            usage = dict(getattr(self.encoder, "last_usage", {}))
            return {
                "status": "ok", "selected_option_id": selected,
                "raw_confidence": float(np.asarray(scores)[int(np.argmax(probabilities))]),
                "probabilities": dict(zip(ids, map(float, probabilities))), "causes_cited": [],
                "input_tokens": usage.get("input_tokens", usage.get("content_tokens", 0)), "output_tokens": 0,
                "api_cost_usd": 0.0, "gpu_seconds": elapsed if device.startswith("cuda") else 0.0,
                "metadata": {"mechanism": self.name, "feature_provenance": dict(self.feature_identity),
                             "feature_usage": usage, "fit": self.fit_metadata,
                             "calibration": self.calibration_metadata,
                             "temperature": self.temperature,
                             "calibration_parameter_fitted": bool(self.calibration_metadata.get("fitted")),
                             "calibration_on_this_benchmark_established": False,
                             "feature_timing_policy": usage.get("cache_policy", "measured_encoder_call_and_decision"),
                             "encoder_cache_present": hasattr(self.encoder, "cache"),
                             "encoder_cache_entries": len(getattr(self.encoder, "cache", {}))},
            }
        except Exception as error:
            return {"status": "error", "selected_option_id": None, "probabilities": {}, "raw_confidence": None,
                    "causes_cited": [], "input_tokens": 0, "output_tokens": 0,
                    "api_cost_usd": 0.0,
                    "gpu_seconds": (time.perf_counter() - started
                                    if self.feature_identity.get("device", "cpu").startswith("cuda") else 0.0),
                    "error": f"{type(error).__name__}: {error}",
                    "metadata": {"mechanism": self.name, "feature_provenance": dict(self.feature_identity)}}


class EmbeddingSimilarity(_FeatureDecision):
    name = "frozen_embedding_cosine_with_validation_temperature"

    def _scores(self, task):
        ids, _, cosine = self._features(task)
        return ids, cosine

    def calibrate(self, val_tasks, oracles):
        val_tasks = list(val_tasks)
        self._check_calibration(val_tasks)
        if not val_tasks:
            raise ValueError("Temperature fitting requires nonempty validation data")
        examples = []
        for task in val_tasks:
            ids, scores = self._scores(task)
            valid = _oracle(oracles, task)
            # A set-valued correct remedy is scored by its total probability.
            examples.append((scores, np.array([option in valid for option in ids])))
        grid = np.logspace(-2, 2, 161)
        losses = [float(np.mean([-math.log(max(float(_softmax(scores, temperature)[mask].sum()), 1e-15))
                                 for scores, mask in examples])) for temperature in grid]
        optimum = min(range(len(grid)), key=lambda index: (losses[index], abs(math.log(grid[index]))))
        self.temperature = float(grid[optimum])
        self.calibration_metadata = {
            "fitted": True, "parameter": "single_softmax_temperature",
            "fitting_scope": "validation_only", "validation_tasks": len(val_tasks),
            "validation_groups": len({_group(task) for task in val_tasks}),
            "validation_task_id_hash": _signature(val_tasks),
            "objective": "mean_negative_log_acceptable_probability_mass",
            "temperature": self.temperature, "validation_loss": losses[optimum],
            "search": {"minimum": .01, "maximum": 100.0, "log_grid_points": len(grid)},
        }
        return self


class LinearProbeDecisionModel(_FeatureDecision):
    name = "logistic_probe_on_frozen_candidate_features"

    def __init__(self, encoder=None, *, regularization=1.0, random_seed=42):
        super().__init__(encoder)
        self.regularization, self.random_seed = float(regularization), int(random_seed)
        self.classifier = None

    def fit(self, train_tasks, oracles):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        train_tasks = list(train_tasks)
        if not train_tasks:
            raise ValueError("Linear probe requires training labels")
        self._record_training(train_tasks)
        features, labels, weights = [], [], []
        group_size = defaultdict(int)
        for task in train_tasks:
            group_size[_group(task)] += 1
        for task in train_tasks:
            ids, matrix, _ = self._features(task)
            positive = _oracle(oracles, task)
            features.extend(matrix)
            labels.extend(int(option in positive) for option in ids)
            # Equal total weight per independent scenario group, then per task.
            weights.extend([1 / (group_size[_group(task)] * len(ids))] * len(ids))
        labels = np.asarray(labels)
        if len(set(labels.tolist())) != 2:
            raise ValueError("Candidate probe requires positive and negative training labels")
        self.scaler = StandardScaler()
        matrix = self.scaler.fit_transform(np.asarray(features), sample_weight=np.asarray(weights))
        self.classifier = LogisticRegression(C=self.regularization, max_iter=1000,
                                             random_state=self.random_seed)
        self.classifier.fit(matrix, labels, sample_weight=np.asarray(weights))
        self.fit_metadata.update(candidate_rows=len(labels), positive_rows=int(labels.sum()),
                                 feature_dimension=matrix.shape[1], regularization_C=self.regularization,
                                 scaler_scope="train_only", weighting="equal_scenario_group_then_task_then_candidate")
        return self

    def _scores(self, task):
        if self.classifier is None:
            raise RuntimeError("Linear probe has not been fitted")
        ids, matrix, _ = self._features(task)
        return ids, self.classifier.decision_function(self.scaler.transform(matrix))


class KNNDecisionModel(_FeatureDecision):
    name = "knn_on_frozen_candidate_features"

    def __init__(self, encoder=None, *, neighbors=5):
        super().__init__(encoder)
        self.neighbors, self.classifier = int(neighbors), None
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")

    def fit(self, train_tasks, oracles):
        from sklearn.neighbors import KNeighborsClassifier

        train_tasks = list(train_tasks)
        if not train_tasks:
            raise ValueError("kNN requires training labels")
        self._record_training(train_tasks)
        features, labels = [], []
        seen = set()
        for task in train_tasks:
            ids, matrix, _ = self._features(task)
            positive = _oracle(oracles, task)
            for option, vector in zip(ids, matrix):
                label = int(option in positive)
                key = (hashlib.sha256(vector.tobytes()).hexdigest(), label)
                if key in seen:
                    continue
                seen.add(key)
                features.append(vector)
                labels.append(label)
        if len(set(labels)) != 2:
            raise ValueError("Candidate kNN requires positive and negative training labels")
        self.classifier = KNeighborsClassifier(n_neighbors=min(self.neighbors, len(labels)),
                                               weights="distance", metric="cosine", algorithm="brute")
        self.classifier.fit(np.asarray(features), np.asarray(labels))
        self.fit_metadata.update(candidate_rows=len(labels), neighbors=self.classifier.n_neighbors,
                                 feature_dimension=len(features[0]), duplicate_feature_label_rows_removed=True)
        return self

    def _scores(self, task):
        if self.classifier is None:
            raise RuntimeError("kNN has not been fitted")
        ids, matrix, _ = self._features(task)
        positive_index = list(self.classifier.classes_).index(1)
        probability = self.classifier.predict_proba(matrix)[:, positive_index]
        # Convert independent binary candidate estimates to relative menu scores.
        probability = np.clip(probability, 1e-6, 1 - 1e-6)
        return ids, np.log(probability) - np.log1p(-probability)


def label_efficiency_curve(model_factory, train_tasks, train_oracles, validation_tasks,
                           validation_oracles, test_tasks, test_oracles, *,
                           budgets=(2, 4, 8, 16), seeds=(0, 1, 2)):
    """Re-fit on nested budgets of independent training scenario groups.

    All wording variants/cases within a selected group move together. Budgets
    count independent groups; actual question and candidate label counts remain
    explicit. Test labels are used only after predictions for scoring.
    """
    train_tasks, validation_tasks, test_tasks = map(list, (train_tasks, validation_tasks, test_tasks))
    _split_guard(train_tasks, "train")
    _split_guard(validation_tasks, "calibration")
    groups = defaultdict(list)
    for task in train_tasks:
        groups[_group(task)].append(task)
    train_groups, validation_groups, test_groups = map(
        lambda tasks: {_group(task) for task in tasks}, (train_tasks, validation_tasks, test_tasks))
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise ValueError("Label-efficiency train/validation/test scenario groups overlap")
    if not groups or not test_tasks:
        raise ValueError("Label-efficiency evaluation requires training groups and test tasks")
    results = []
    for seed in seeds:
        order = sorted(groups)
        np.random.default_rng(seed).shuffle(order)
        for budget in budgets:
            if not isinstance(budget, int) or budget < 1:
                raise ValueError("Label budgets must be positive integer group counts")
            selected = set(order[:budget])
            subset = [task for group in order if group in selected for task in groups[group]]
            model = model_factory()
            model.fit(subset, train_oracles)
            model.calibrate(validation_tasks, validation_oracles)
            correct = valid = 0
            for task in test_tasks:
                prediction = model.predict(solver_view(task))
                valid += prediction.get("status") == "ok"
                correct += (prediction.get("status") == "ok"
                            and prediction.get("selected_option_id") in _oracle(test_oracles, task))
            results.append({"seed": int(seed), "requested_group_budget": budget,
                            "actual_group_budget": len(selected), "question_labels": len(subset),
                            "budget_unit": "independent_scenario_groups", "test_tasks": len(test_tasks),
                            "accuracy": correct / len(test_tasks), "valid_response_rate": valid / len(test_tasks),
                            "training_task_id_hash": _signature(subset),
                            "feature_provenance": dict(model.feature_identity)})
    return results
