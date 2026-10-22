"""Offline Hugging Face and compute-local Ollama decision adapters.

No adapter downloads a model, invokes a hosted service, truncates input, or
silently substitutes a model or probability readout. Heavy imports are lazy.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import ipaddress
import json
import math
import os
from pathlib import Path
import random
import time
import urllib.parse
import urllib.request

METHODS = {"choice", "logprobs", "verbalized", "sampling"}


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("Model loading and inference require a Slurm compute-node allocation")


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _hf_context_extension(spec, configuration):
    """Clone the released Qwen2.5 config for the explicit official YaRN variant.

    Checkpoint files/weights stay untouched. Other models and default loads keep
    their original configuration. The operational cap is deliberately65536,
    below the documented131072 extended positional configuration.
    """
    extension = spec.get('context_extension')
    if extension is None:
        return configuration, None
    if extension != 'qwen25-yarn-v1':
        raise ValueError('Unknown explicit Hugging Face context extension')
    identity = str(spec.get('model_path', '')) + ' ' + str(getattr(configuration, '_name_or_path', ''))
    if (getattr(configuration, 'model_type', None) != 'qwen2'
            or 'qwen2.5' not in identity.lower()
            or getattr(configuration, 'max_position_embeddings', None) != 32768
            or getattr(configuration, 'rope_scaling', None) is not None):
        raise ValueError('qwen25-yarn-v1 requires the unextended released Qwen2.5 qwen2 configuration')
    requested = _positive_int(spec.get('max_context', 65536), 'max_context')
    if requested > 65536:
        raise ValueError('qwen25-yarn-v1 operational max_context cannot exceed65536')
    original = deepcopy(configuration.to_dict())
    effective = deepcopy(configuration)
    effective.rope_scaling = {'factor': 4.0, 'original_max_position_embeddings': 32768, 'type': 'yarn'}
    # Qwen2Config normally normalizes this legacy release key in __init__.
    # We clone in memory, so explicitly supply the current equivalent alias.
    effective.rope_scaling['rope_type'] = 'yarn'
    effective.max_position_embeddings = 131072
    return effective, {'protocol': extension, 'checkpoint_files_modified': False,
        'weights_modified': False, 'training_performed': False,
        'release_configuration': 'Staged Qwen2.5 README Processing Long Texts: factor4.0, original32768, typeyarn.',
        'original_config': original, 'original_config_sha256': _json_sha256(original),
        'effective_config': deepcopy(effective.to_dict()),
        'effective_config_sha256': _json_sha256(effective.to_dict()),
        'original_rope_scaling': deepcopy(getattr(configuration, 'rope_scaling', None)),
        'effective_rope_scaling': deepcopy(effective.rope_scaling),
        'release_rope_scaling': {'factor': 4.0, 'original_max_position_embeddings': 32768, 'type': 'yarn'},
        'rope_type_alias': 'Transformers canonical rope_type=yarn is identical to the release type=yarn.',
        'original_max_position_embeddings': 32768,
        'effective_max_position_embeddings': 131072, 'operational_max_context': requested}


def _official_logits(model, inputs):
    """Keep model-specific logit transforms in the official causal-LM forward.

    Gemma's final softcap, for example, runs after its output embedding head.
    A base-model-plus-head reconstruction would silently skip that processing.
    Older models without a documented last-logit argument use their complete
    forward output; this costs memory but preserves the released computation.
    """
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        parameters = {}
    options = {"use_cache": False, "return_dict": True}
    limit = "full_forward_output"
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            options[name] = 1
            limit = name
            break
    return model(**inputs, **options).logits, limit


def _generation_kwargs(*, sample, max_new_tokens, pad_token_id, temperature):
    # Qwen's staged generation_config enables a repetition penalty of 1.05.
    # Override it for every readout, including greedy generation, so the HF
    # and Ollama adapters share the declared generation policy.
    return {"do_sample": sample, "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id, "repetition_penalty": 1.0,
            "top_k": 0, "top_p": 1.0, "temperature": temperature if sample else 1.0}


def _local_path(value, name):
    if not value:
        raise ValueError(f"{name} must name an already staged local directory")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {name}: {path}; no automatic download")
    return path


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    result = []
    for message in messages:
        if (not isinstance(message, dict) or message.get("role") not in
                {"system", "user", "assistant"} or not isinstance(message.get("content"), str)):
            raise ValueError("Each message requires a supported role and string content")
        result.append({"role": message["role"], "content": message["content"]})
    return result


def _validate_ids(option_ids):
    if (not isinstance(option_ids, list) or not option_ids or
            any(not isinstance(key, str) or not key.strip() for key in option_ids) or
            len(set(option_ids)) != len(option_ids)):
        raise ValueError("option_ids must be unique nonempty strings")
    return list(option_ids)


def _instruction(option_ids, method, *, cite_causes=False):
    labels = json.dumps(option_ids, ensure_ascii=False)
    if method == "verbalized":
        if cite_causes:
            return (f"Choose exactly one offered option identifier from {labels}. Return ONLY a JSON object "
                    'with keys "selected_option_id", "confidence", and "causes_cited". Confidence is your probability '
                    "that the selected option is correct, between 0 and 1. causes_cited must be an array of distinct "
                    "public reading/state identifiers, public precondition identifiers or schema paths, or public error "
                    "codes that support your decision. Cite exact identifiers present in the supplied public evidence; "
                    "do not invent identifiers. Use an empty array if no relevant identifier is available. Do not add explanations.")
        return (f"Choose exactly one offered option identifier from {labels}. Return ONLY a JSON object "
                'with keys "selected_option_id" and "confidence". Confidence is your probability '
                "that the selected option is correct, between 0 and 1. Do not add explanations.")
    return f"Reply with exactly one option identifier from {labels}. Do not include any other text."


def _messages(messages, option_ids, method, *, cite_causes=False):
    result = _validate_messages(messages)
    instruction = _instruction(option_ids, method, cite_causes=cite_causes)
    readout = ("\n\nCurrent decision readout (overrides earlier response-format instructions only): "
               + instruction)
    system = next((message for message in result if message["role"] == "system"), None)
    if system is None:
        result.insert(0, {"role": "system", "content": readout.strip()})
    else:
        system["content"] += readout
    # Append instructions to the last user turn: do not discard or compact any
    # observation, manual, action, correction, or prior result supplied by caller.
    if result[-1]["role"] == "user":
        result[-1]["content"] += "\n\n" + instruction
    else:
        result.append({"role": "user", "content": instruction})
    return result


def _parse_label(raw, option_ids):
    value = raw.strip()
    if value not in option_ids:
        raise ValueError("Completion is not exactly one offered option identifier")
    return value


def _parse_verbalized(raw, option_ids):
    answer = json.loads(raw)
    if not isinstance(answer, dict) or answer.get("selected_option_id") not in option_ids:
        raise ValueError("Invalid verbalized-confidence JSON or option identifier")
    value = answer.get("confidence")
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Confidence must be a finite number between zero and one")
    return answer["selected_option_id"], float(value)


def _citation_array(value):
    if (not isinstance(value, list) or len(value) > 64
            or any(not isinstance(item, str) or not item.strip() or len(item) > 512 for item in value)
            or len(set(value)) != len(value)):
        raise ValueError("Cited verbalized answers require a distinct string array causes_cited")
    return value


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return value


def _completion_decision(raw, count, generated, ids, method, seed, *, cite_causes=False):
    result = {"selected_option_id": None, "confidence": None, "probabilities": None,
              "input_tokens": count, "output_tokens": generated,
              "metadata": {"raw_completion": raw, "seed": seed,
                           "probability_semantics": "selected_option_self_reported_confidence_only"
                           if method == "verbalized" else "none"}}
    try:
        if method == "verbalized":
            selected, confidence = _parse_verbalized(raw, ids)
            if cite_causes:
                result["causes_cited"] = _citation_array(json.loads(raw).get("causes_cited"))
                result["metadata"]["citation_protocol"] = "explicit_public_identifier_array-v1"
        else:
            selected, confidence = _parse_label(raw, ids), None
        result.update(selected_option_id=selected, confidence=confidence)
    except (ValueError, TypeError) as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}",
                      error_type="model_output_invalid")
        result["metadata"].update(error_type="model_output_invalid", error_kind="model_output")
    return result


class DecisionBackend:
    """Canonical public interface: decide(messages, option_ids, **readout)."""

    def __init__(self, spec):
        self.spec = dict(spec)
        if type(spec.get("cite_causes", False)) is not bool:
            raise ValueError("cite_causes must be boolean")
        self.default_method = spec.get("method", "choice")
        if self.default_method not in METHODS:
            raise ValueError(f"Unknown decision method: {self.default_method}")
        self.identity = {}

    def decide(self, messages, option_ids, *, option_descriptions=None,
               method=None, samples=10, seed=0, temperature=0.8):
        require_slurm()
        ids = _validate_ids(option_ids)
        method = method or self.default_method
        if method not in METHODS:
            raise ValueError(f"Unknown decision method: {method}")
        cited = self.spec.get("cite_causes", False)
        if cited and method != "verbalized":
            raise ValueError("cite_causes is a separately declared verbalized-only variant")
        if option_descriptions is not None and set(option_descriptions) != set(ids):
            raise ValueError("option_descriptions must cover the exact offered identifiers")
        if method == "sampling":
            if isinstance(samples, bool) or not isinstance(samples, int) or samples < 10:
                raise ValueError("Sampling requires at least ten independent draws")
            if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
                    not math.isfinite(temperature) or temperature <= 0):
                raise ValueError("Sampling temperature must be finite and positive")
        seed = _seed(seed)
        prepared = _messages(messages, ids, method, cite_causes=cited)
        started = time.perf_counter()
        metadata = {**self.identity, "method": method, "option_count": len(ids),
                    "messages_sha256": hashlib.sha256(json.dumps(
                        prepared, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                    "context_policy": "error_without_truncation",
                    "calibration_on_this_benchmark_established": False}
        if cited:
            metadata["citation_protocol"] = "explicit_public_identifier_array-v1"
        try:
            result = self._decide(prepared, ids, method, samples, seed, temperature)
            if cited and result.get("status", "ok") == "ok":
                try:
                    _citation_array(result.get("causes_cited"))
                except ValueError as error:
                    result.update(status="error", selected_option_id=None, confidence=None,
                                  probabilities=None, causes_cited=[], error=str(error),
                                  error_type="model_output_invalid")
                    result.setdefault("metadata", {}).update(error_type="model_output_invalid", error_kind="model_output")
            return {"status": "ok", "selected_option_id": None, "probabilities": None,
                    "confidence": None, "causes_cited": [], "input_tokens": None,
                    "output_tokens": None, "api_cost_usd": 0.0, **result,
                    "latency_s": time.perf_counter() - started,
                    "metadata": {**metadata, **result.get("metadata", {})}}
        except Exception as error:
            return {"status": "error", "selected_option_id": None, "probabilities": None,
                    "confidence": None, "causes_cited": [], "input_tokens": None,
                    "output_tokens": None, "api_cost_usd": 0.0,
                    "latency_s": time.perf_counter() - started,
                    "error": f"{type(error).__name__}: {error}",
                    "error_type": "infrastructure_error",
                    "metadata": {**metadata, "error_type": "infrastructure_error",
                                 "error_kind": "infrastructure"}}

    def choose(self, messages, labels, *, request_id=None, **kwargs):
        """Compatibility view; all model work goes through canonical decide."""
        result = self.decide(messages, labels, **kwargs)
        probabilities = result.get("probabilities")
        return {**result,
                "index": labels.index(result["selected_option_id"])
                if result.get("selected_option_id") in labels else None,
                "probabilities": [probabilities[label] for label in labels]
                if probabilities is not None else None,
                "probability_semantics": result.get("metadata", {}).get("probability_semantics"),
                "raw_response": result.get("metadata", {}).get("raw_completion"),
                "usage": {"input_tokens": result.get("input_tokens"),
                          "output_tokens": result.get("output_tokens")},
                "elapsed_s": result.get("latency_s"), "request_id": request_id}


class HuggingFaceBackend(DecisionBackend):
    def __init__(self, spec):
        require_slurm()
        super().__init__({"method": "logprobs", **spec})
        path = _local_path(spec.get("model_path"), "model_path")
        self.device = spec.get("device", "cuda:0")
        self.max_new_tokens = _positive_int(spec.get("max_new_tokens", 192), "max_new_tokens")
        extension = spec.get('context_extension')
        if extension not in (None, 'qwen25-yarn-v1'):
            raise ValueError('Unknown explicit Hugging Face context extension')
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable; no CPU substitution")
        self.torch = torch
        dtype_name = spec.get("dtype", "bfloat16" if self.device.startswith("cuda") else "float32")
        if dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        started = time.perf_counter()
        extension_details, model_options = None, {}
        if extension is not None:
            original_config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
            effective_config, extension_details = _hf_context_extension(spec, original_config)
            from transformers.modeling_rope_utils import rope_config_validation
            rope_config_validation(effective_config)
            model_options['config'] = effective_config
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                                      trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, dtype_name), attn_implementation="sdpa", **model_options).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if not isinstance(maximum, int) or maximum < 1:
            raise ValueError("Model configuration has no finite max_position_embeddings")
        requested_context = spec.get("max_context", 65536 if extension is not None else None)
        self.max_context = min(_positive_int(maximum if requested_context is None else requested_context,
                                            "max_context"), maximum)
        weights = sorted(set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin")))
        if not weights:
            raise FileNotFoundError("Local snapshot has no recognized weight files")
        provenance = path / "benchmark-model-manifest.json"
        if not provenance.exists():
            provenance = path / "comparison-model-manifest.json"
        generation_source = path / "generation_config.json"
        loaded_generation_config = self.model.generation_config.to_dict()
        self.identity = {
            "backend": "hf", "model_path": str(path), "device": self.device,
            "dtype": dtype_name, "max_context": self.max_context,
            "max_new_tokens": self.max_new_tokens, "weights_frozen": True,
            "local_files_only": True, "trust_remote_code": False,
            "model_type": self.model.config.model_type,
            "weight_sha256": {item.name: _sha256(item) for item in weights},
            "config_sha256": _sha256(path / "config.json"),
            "generation_config_file_sha256": _sha256(generation_source)
            if generation_source.is_file() else None,
            "loaded_generation_config": loaded_generation_config,
            "loaded_generation_config_sha256": _json_sha256(loaded_generation_config),
            "generation_policy": {"repetition_penalty": 1.0, "top_k": 0, "top_p": 1.0,
                                  "greedy_temperature": 1.0,
                                  "other_generation_defaults": "recorded loaded_generation_config"},
            "tokenizer_sha256": {item.name: _sha256(item) for item in sorted(path.iterdir())
                                 if item.is_file() and ("tokenizer" in item.name or
                                     item.suffix in {".jinja", ".model"} or
                                     item.name in {"vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"})},
            "loaded_chat_template_sha256": _json_sha256(self.tokenizer.chat_template),
            "download_manifest": json.loads(provenance.read_text()) if provenance.exists() else None,
            "torch": torch.__version__, "transformers": _version("transformers"),
            "load_wall_time_s": time.perf_counter() - started,
        }
        if extension_details is not None:
            self.identity.update(context_extension=extension,
                context_extension_details=extension_details,
                effective_model_config=deepcopy(self.model.config.to_dict()),
                effective_model_config_sha256=_json_sha256(self.model.config.to_dict()),
                effective_rope_scaling=deepcopy(self.model.config.rope_scaling))

    def _inputs(self, messages, reserve):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt",
                                truncation=False).to(self.device)
        count = int(inputs.input_ids.shape[-1])
        if count + reserve > self.max_context:
            raise ValueError(f"Public input {count} + output reserve {reserve} exceeds "
                             f"context {self.max_context}; evidence was not truncated")
        return inputs, count

    def _synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def _scores(self, messages, ids):
        encoded = [self.tokenizer.encode(key, add_special_tokens=False) for key in ids]
        if any(len(tokens) != 1 for tokens in encoded) or len({tokens[0] for tokens in encoded}) != len(ids):
            raise ValueError("The conditional next-token readout requires unique single-token option identifiers")
        inputs, count = self._inputs(messages, 1)
        with self.torch.inference_mode():
            output_logits, limit = _official_logits(self.model, inputs)
            logits = output_logits[0, -1, [tokens[0] for tokens in encoded]].float()
            scores = logits.softmax(-1).cpu().tolist()
        self.last_score_receipt = {"forward_logit_limit": limit,
                                   "model_class": type(self.model).__name__,
                                   "logit_source": "official_causal_lm_forward"}
        self._synchronize()
        return dict(zip(ids, scores)), count

    def _generate(self, messages, *, sample=False, seed=0, temperature=0.8,
                  prefix_allowed_tokens_fn=None):
        inputs, count = self._inputs(messages, self.max_new_tokens)
        devices = [self.torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        # fork_rng preserves unrelated caller RNG state; each sampling request
        # gets a distinct deterministic seed recorded with the raw completion.
        with self.torch.random.fork_rng(devices=devices), self.torch.inference_mode():
            self.torch.manual_seed(seed)
            kwargs = _generation_kwargs(sample=sample, max_new_tokens=self.max_new_tokens,
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        temperature=temperature)
            effective = {**self.model.generation_config.to_dict(), **kwargs}
            controls = {} if prefix_allowed_tokens_fn is None else {'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn}
            generated = self.model.generate(**inputs, **kwargs, **controls)
        suffix = generated[0, count:]
        self._synchronize()
        self.last_generation_receipt = {"input_tokens": count, "output_tokens": int(suffix.numel()),
                                        "generation_kwargs": kwargs,
                                        "effective_generation_config": effective,
                                        "effective_generation_config_sha256": _json_sha256(effective)}
        if prefix_allowed_tokens_fn is not None:
            self.last_generation_receipt['structured_output'] = prefix_allowed_tokens_fn.receipt()
        return self.tokenizer.decode(suffix, skip_special_tokens=True), count, int(suffix.numel())

    def _decide(self, messages, ids, method, samples, seed, temperature):
        if method == "logprobs":
            probabilities, count = self._scores(messages, ids)
            selected = max(ids, key=probabilities.get)
            return {"selected_option_id": selected, "probabilities": probabilities,
                    "confidence": probabilities[selected], "input_tokens": count, "output_tokens": 0,
                    "metadata": {"probability_semantics": "next_token_softmax_conditional_on_offered_menu",
                                 "conditional_on_valid_labels": True,
                                 "generated_answer_tokens": 0, **self.last_score_receipt}}
        if method == "sampling":
            # Actual repeated generation, matching the Ollama sampling protocol;
            # this is not sampling from a previously normalized menu softmax.
            return _sample_completions(self._generate, messages, ids, samples, seed, temperature)
        raw, count, generated = self._generate(messages, seed=seed)
        result = _completion_decision(raw, count, generated, ids, method, seed,
                                      cite_causes=getattr(self, "spec", {}).get("cite_causes", False))
        result["metadata"]["generation"] = dict(self.last_generation_receipt)
        return result


def _sample_completions(generate, messages, ids, samples, seed, temperature):
    seed_generator = random.Random(seed)
    draws, raw_draws, invalid = [], [], []
    input_tokens = output_tokens = 0
    usage_known = True
    for index in range(samples):
        draw_seed = seed_generator.randrange(2**31)
        entry = {"index": index, "seed": draw_seed}
        try:
            raw, input_count, output_count = generate(messages, sample=True,
                                                     seed=draw_seed, temperature=temperature)
            input_tokens += input_count
            output_tokens += output_count
            entry.update(raw_completion=raw, input_tokens=input_count, output_tokens=output_count)
            owner = getattr(generate, "__self__", None)
            if getattr(owner, "last_generation_receipt", None) is not None:
                entry["provider_usage"] = dict(owner.last_generation_receipt)
            selected = _parse_label(raw, ids)
            draws.append(selected)
            entry["selected_option_id"] = selected
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            if "raw_completion" not in entry:
                usage_known = False
            invalid.append(entry)
        raw_draws.append(entry)
    counts = Counter(draws)
    metadata = {"probability_semantics": "empirical_unconstrained_completion_frequency",
                "samples": samples, "seed": seed, "temperature": temperature,
                "draws": raw_draws, "valid_draws": len(draws), "invalid_draws": len(invalid),
                "prefill_shared": False, "reasoning_ensemble": False,
                "sampling_settings": {"top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}}
    if draws:
        winning = max(ids, key=lambda key: counts[key])
        metadata.update(winning_option_id=winning, winning_vote_fraction=counts[winning] / samples)
    if invalid:
        # Renormalizing only valid draws would hide parse failures and make
        # this method incomparable to strict single-completion evaluation.
        error_type = "model_output_invalid" if usage_known else "infrastructure_error"
        metadata.update(error_type=error_type,
                        error_kind="model_output" if usage_known else "infrastructure")
        return {"status": "error", "selected_option_id": None, "probabilities": None,
                "confidence": None, "error": f"{len(invalid)}/{samples} sampled completions were invalid",
                "error_type": error_type,
                "input_tokens": input_tokens if usage_known else None,
                "output_tokens": output_tokens if usage_known else None, "metadata": metadata}
    probabilities = {key: counts[key] / samples for key in ids}
    selected = max(ids, key=probabilities.get)
    return {"selected_option_id": selected, "probabilities": probabilities,
            "confidence": probabilities[selected], "input_tokens": input_tokens,
            "output_tokens": output_tokens, "metadata": metadata}


def _loopback_url(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Local server URL must be HTTP(S) without credentials, query, or fragment")
    host = parsed.hostname
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("Only a loopback model server is allowed")
        except (TypeError, ValueError):
            raise ValueError("Only a loopback model server is allowed") from None
    return value.rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Model server redirects are refused")


class OllamaBackend(DecisionBackend):
    """Ollama raw generation with matching local tokenizer context checks.

    /api/chat may silently compact an overflowing conversation. This adapter
    formats the complete chat locally and sends /api/generate with raw=true,
    then verifies the server consumed the exact expected number of tokens.
    """

    def __init__(self, spec):
        require_slurm()
        super().__init__(spec)
        base_url = spec.get("base_url", os.environ.get("COMPARISON_OLLAMA_URL", "http://127.0.0.1:11434"))
        if base_url.startswith("env:"):
            variable = base_url[4:]
            if not os.environ.get(variable):
                raise ValueError(f"Missing local server URL environment variable {variable}")
            base_url = os.environ[variable]
        self.base_url = _loopback_url(base_url)
        self.model_name = spec.get("model")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("Ollama requires an explicit already staged model tag")
        self.timeout = float(spec.get("timeout_s", 600))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        models = self._request("/api/tags").get("models", [])
        entry = next((item for item in models if self.model_name in
                      {item.get("name"), item.get("model")}), None)
        if entry is None or not entry.get("digest"):
            raise FileNotFoundError(f"Ollama model {self.model_name!r} is not staged; no automatic pull")
        show = self._request("/api/show", {"model": self.model_name})
        declared = [value for key, value in show.get("model_info", {}).items()
                    if key.endswith(".context_length") and isinstance(value, int) and value > 0]
        if not declared:
            raise ValueError("Ollama model metadata has no declared context length")
        self.max_context = min(_positive_int(spec.get("max_context", min(declared)), "max_context"),
                               min(declared))
        self.max_new_tokens = _positive_int(spec.get("max_new_tokens", 192), "max_new_tokens")
        tokenizer_path = _local_path(spec.get("tokenizer_path"), "tokenizer_path")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True,
                                                      trust_remote_code=False)
        self.identity = {"backend": "ollama", "model": self.model_name,
                         "model_digest": entry["digest"], "base_url": self.base_url,
                         "server_version": self._request("/api/version").get("version"),
                         "model_details": entry.get("details"), "max_context": self.max_context,
                         "max_new_tokens": self.max_new_tokens,
                         "tokenizer_path": str(tokenizer_path),
                         "tokenizer_sha256": {item.name: _sha256(item) for item in sorted(tokenizer_path.iterdir())
                                              if item.is_file() and ("tokenizer" in item.name or
                                                  item.suffix in {".jinja", ".model"} or
                                                  item.name in {"vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"})},
                         "loaded_chat_template_sha256": _json_sha256(self.tokenizer.chat_template),
                         "template": "local_tokenizer_chat_template_raw_generate",
                         "local_files_only": True, "transformers": _version("transformers")}

    def _request(self, endpoint, payload=None):
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def _generate(self, messages, *, sample=False, seed=0, temperature=0.8):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        count = len(self.tokenizer.encode(prompt, add_special_tokens=False, truncation=False))
        if count + self.max_new_tokens > self.max_context:
            raise ValueError(f"Public input {count} + output reserve {self.max_new_tokens} exceeds "
                             f"context {self.max_context}; evidence was not truncated")
        result = self._request("/api/generate", {
            "model": self.model_name, "prompt": prompt, "raw": True, "stream": False,
            "keep_alive": "30m", "options": {"num_ctx": self.max_context,
                "num_predict": self.max_new_tokens, "temperature": temperature if sample else 0,
                "seed": seed, "top_k": 0, "top_p": 1.0, "repeat_penalty": 1.0}})
        if result.get("model") != self.model_name:
            raise ValueError("Ollama response model does not match requested model tag")
        if result.get("done") is not True:
            raise ValueError("Ollama returned an incomplete generation")
        if result.get("prompt_eval_count") != count:
            raise ValueError(f"Ollama consumed {result.get('prompt_eval_count')} prompt tokens, expected "
                             f"{count}; possible tokenizer mismatch or server truncation")
        generated = result.get("eval_count")
        if isinstance(generated, bool) or not isinstance(generated, int) or generated < 0:
            raise ValueError("Ollama response lacks actual generated-token usage")
        if count + generated > self.max_context:
            raise ValueError("Ollama generation exceeded the validated context budget")
        raw = result.get("response")
        if not isinstance(raw, str):
            raise ValueError("Ollama response text is missing")
        self.last_generation_receipt = {key: result.get(key) for key in
            ("prompt_eval_count", "eval_count", "total_duration", "load_duration",
             "prompt_eval_duration", "eval_duration", "done_reason")}
        return raw, count, generated

    def _decide(self, messages, ids, method, samples, seed, temperature):
        if method == "logprobs":
            raise NotImplementedError("This Ollama adapter does not implement a complete conditional-menu "
                                      "logprob readout; use choice, verbalized, or sampling")
        # Confirm the tag still resolves to the recorded digest before inference.
        models = self._request("/api/tags").get("models", [])
        entry = next((item for item in models if self.model_name in
                      {item.get("name"), item.get("model")}), None)
        if entry is None or entry.get("digest") != self.identity["model_digest"]:
            raise ValueError("Ollama model tag changed after backend initialization")
        if method == "sampling":
            return _sample_completions(self._generate, messages, ids, samples, seed, temperature)
        raw, count, generated = self._generate(messages, seed=seed)
        result = _completion_decision(raw, count, generated, ids, method, seed,
                                      cite_causes=getattr(self, "spec", {}).get("cite_causes", False))
        result["metadata"]["provider_usage"] = dict(self.last_generation_receipt)
        return result


def build_backend(config):
    if not isinstance(config, dict):
        raise ValueError("Backend configuration must be an object")
    kind = config.get("kind", config.get("backend"))
    if kind in {"hf", "huggingface"}:
        return HuggingFaceBackend(config)
    if kind == "ollama":
        return OllamaBackend(config)
    if kind == "native":
        from .native import NativeBackend
        return NativeBackend(config)
    if kind == "features":
        from .features import FrozenFeatureBackend
        return FrozenFeatureBackend(config)
    if kind == "reference":
        from .features import ReferenceBackend
        return ReferenceBackend(config)
    raise ValueError(f"Unsupported local backend: {kind!r}")


create_backend = build_backend
