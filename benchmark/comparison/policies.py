"""Public-input adapters; the released native evaluator owns all grades.

Sequential control deliberately inherits the release's ordering, hierarchy,
manual projection, full history, and readable evidence rendering. Only model
selection at an offered hierarchy node is replaced.
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time


_BUNDLE = None
_ACTIVE = None


class InvalidModelDecision(ValueError):
    """Malformed model output: the native grader scores this proposal wrong."""


def load_bundle(bundle):
    global _BUNDLE
    bundle = Path(bundle).resolve()
    if _BUNDLE is not None:
        if _BUNDLE[0] != bundle:
            raise RuntimeError("A process cannot mix different instrument releases")
        return _BUNDLE[1]
    location = bundle / "eval/__init__.py"
    if not location.is_file():
        raise ValueError(f"No instrument evaluator at {bundle}")
    name = "_comparison_instrument_eval"
    spec = importlib.util.spec_from_file_location(name, location,
                                                submodule_search_locations=[str(location.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    helpers = {part: importlib.import_module(f"{name}.{part}")
               for part in ("common", "prompts", "mcqa", "sequential", "validate")}
    _BUNDLE = bundle, helpers
    return helpers


def validated_result(result, labels):
    """Reject malformed backend choices or invented probability vectors."""
    if not isinstance(result, dict):
        raise ValueError("Backend result must be an object")
    if result.get("status") != "ok":
        raise RuntimeError(result.get("error", "Backend did not return an ok choice"))
    if result.get("selected_option_id") not in labels:
        raise ValueError("Backend did not select an offered label")
    import math
    confidence = result.get("confidence")
    if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                                   or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("Backend confidence must be finite and between zero and one")
    probabilities = result.get("probabilities")
    if probabilities is not None:
        if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
            raise ValueError("Backend probabilities must cover every offered label exactly")
        if not all(not isinstance(p, bool) and isinstance(p, (int, float))
                   and math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("Backend probabilities must be finite and nonnegative")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5, rel_tol=1e-5):
            raise ValueError("Backend probability vector must sum to one")
    return result


def decision_parameters(experiment):
    return {"method": experiment.get("method", "choice"),
            "samples": experiment.get("samples", 10),
            "seed": experiment.get("seed", 0),
            "temperature": experiment.get("temperature", 0.8)}


def configure_sequential(bundle, backend, experiment):
    global _ACTIVE
    _ACTIVE = {"bundle": Path(bundle).resolve(), "backend": backend,
               "experiment": copy.deepcopy(experiment)}


def clear_sequential():
    global _ACTIVE
    _ACTIVE = None


def make_policy(context):
    """Factory consumed by the frozen evaluator; context is public only."""
    if _ACTIVE is None:
        raise RuntimeError("Use comparison.run to configure this policy")
    load_bundle(_ACTIVE["bundle"])
    # The evaluator has already activated its bundled modules, with its
    # mixed-import check and genuine Slurm guard still intact.
    from stepwise_choice.policies import (
        LocalChoicePolicy, LABELS, PROTOCOL, PROJECTION_PROTOCOL, RENDER_PROTOCOL,
        SYSTEM, render_model_payload, fingerprint, ChoiceContextError,
    )
    from stepwise_bench.policies import require_allocation
    backend = _ACTIVE["backend"]
    experiment = copy.deepcopy(_ACTIVE["experiment"])

    class BackendChoicePolicy(LocalChoicePolicy):
        def __init__(self):
            require_allocation()
            self.menu_builder = context["menu_builder"]
            self.seed = experiment.get("menu_seed", 42)
            self.max_context = experiment.get("backend", {}).get("max_context")
            self.identity = {"method": "comparison-public-menu-backend",
                             "backend": copy.deepcopy(backend.identity),
                             "decision_parameters": decision_parameters(experiment),
                             "protocol": PROTOCOL, "projection_protocol": PROJECTION_PROTOCOL,
                             "render_protocol": RENDER_PROTOCOL, "privileged": False,
                             "menu_order_seed": self.seed,
                             "public_preset_bank_sha256": fingerprint(self.menu_builder.bank),
                             "system_prompt_sha256": fingerprint(SYSTEM),
                             "context_overflow_policy": "unscored infrastructure error; never truncate"}
            self.last_action_usage = {}

        def _score(self, choices, payload, stage, selection_context, argument_options=None):
            labels = LABELS[:len(choices)]
            if not labels or len(choices) > len(LABELS):
                raise ValueError("A hierarchy node must contain one to twenty labels")
            displayed = [{"label": label, **copy.deepcopy(choice)}
                         for label, choice in zip(labels, choices)]
            evidence = {**payload, "selection_stage": stage,
                        "selection_context": copy.deepcopy(selection_context), "choices": displayed,
                        "required_reply": "Exactly one label from " + ", ".join(labels)}
            rendered = render_model_payload(evidence, argument_options=argument_options)
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps(rendered, sort_keys=True,
                         ensure_ascii=False, allow_nan=False, separators=(",", ":"))}]
            event = {"stage": stage, "prompt_sha256": fingerprint(messages),
                     "menu_sha256": fingerprint(displayed), "choices": displayed,
                     "selection_context": copy.deepcopy(selection_context),
                     "status": "started", "input_tokens_known": False}
            self.last_action_usage["selection_events"].append(event)
            started = time.perf_counter()
            try:
                result = backend.decide(messages, list(labels),
                    option_descriptions={label: json.dumps(choice, ensure_ascii=False, sort_keys=True,
                                                           allow_nan=False)
                                         for label, choice in zip(labels, choices)},
                    **decision_parameters(experiment))
                event["backend_result"] = copy.deepcopy(result)
                for direction in ("input", "output"):
                    count = result.get(direction + "_tokens")
                    known = type(count) is int and count >= 0
                    event[direction + "_tokens_known"] = known
                    if known:
                        event[direction + "_tokens"] = count
                        self.last_action_usage[direction + "_tokens"] += count
                    else:
                        self.last_action_usage[direction + "_tokens_known"] = False
                if result.get("status") != "ok":
                    if result.get("metadata", {}).get("error_kind") == "model_output":
                        raise InvalidModelDecision(result.get("error", "Malformed model reply"))
                    raise ChoiceContextError(result.get("error", "Backend infrastructure failure"))
                try:
                    validated_result(result, labels)
                except (ValueError, RuntimeError) as error:
                    raise ChoiceContextError(f"Malformed backend result: {error}") from error
                selected = labels.index(result["selected_option_id"])
                event.update(status="ok", chosen_label=labels[selected],
                             label_probabilities=result.get("probabilities"),
                             confidence=result.get("confidence"),
                             probability_semantics=result.get("probability_semantics", result.get("metadata", {}).get("probability_semantics")),
                             selected_choice=copy.deepcopy(choices[selected]))
                # Backend timing is wall time: do not relabel it GPU time.
                self.last_action_usage["gpu_seconds"] = None
                self.last_action_usage["api_cost_usd"] = 0.0
                return selected
            except Exception as error:
                event.update(status="error", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                event["wall_time_s"] = time.perf_counter() - started

    return BackendChoicePolicy()


factory = make_policy
