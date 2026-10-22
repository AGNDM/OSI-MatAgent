"""Local, offline model boundaries for the repository's actual ``app.graph``.

The graph, specialist schemas, public projections, action gate and native lab
remain the app's implementations.  This module adapts only model transport and
readout.  The caller puts the authenticated comparison package and app checkout
on ``sys.path``; no live study source is changed or vendored here.
"""
from __future__ import annotations

from copy import deepcopy
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from comparison.backends import HuggingFaceBackend, build_backend, require_slurm
from app.local_decision import option_prompt
from app.local_roles import ExpertResponseError
from app.models import ActionCandidate, OperationDecision, RouteDecision
from app.operation_decision import DECISION_PROMPT, validate_decision


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _lock(backend):
    # An expert and chooser can share one loaded HF model.  Calls remain serial
    # while generation's time limit is temporarily installed on that model.
    if not hasattr(backend, "_graph_local_lock"):
        backend._graph_local_lock = threading.RLock()
    return backend._graph_local_lock


def _check_shared_identity(spec, backend):
    """A shared object must not silently substitute another declared model."""
    identity = backend.identity
    if spec.get('context_extension') != identity.get('context_extension'):
        raise ValueError('Shared backend context_extension differs from the declared configuration')
    for field in ("model_path", "artifact_path"):
        if field in spec and field in identity and Path(spec[field]).expanduser().resolve() != Path(identity[field]).expanduser().resolve():
            raise ValueError(f"Shared backend {field} differs from the declared local artifact")
    for field in ("device", "dtype", "max_context", "family", "effort"):
        if field in spec and field in identity and spec[field] != identity[field]:
            raise ValueError(f"Shared backend {field} differs from the declared configuration")


def _usage(input_tokens, output_tokens):
    if (isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0
            or isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 0):
        return None
    return {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens}


def _remaining(deadline, timeout_seconds):
    timeout = min(timeout_seconds, deadline - time.monotonic())
    if timeout <= 0:
        raise TimeoutError("Episode deadline expired before local inference")
    return timeout


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant is forbidden: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object field: {key}")
        result[key] = value
    return result


def _fresh_call(purpose):
    """Preparation errors must never inherit a previous request's usage."""
    def decorate(function):
        @wraps(function)
        def call(self, *args, **kwargs):
            require_slurm()
            started = time.monotonic()
            self.last_call = {'backend_id': self.backend_id, 'purpose': purpose,
                              'request_sent': False, 'usage': None,
                              'response_classification': 'input_preparation'}
            if purpose == 'specialist' and args:
                self.last_call['role'] = args[0]
            try:
                return function(self, *args, **kwargs)
            except Exception as error:
                self.last_call['error'] = f'{type(error).__name__}: {error}'
                raise
            finally:
                self.last_call.setdefault('elapsed_ms', (time.monotonic() - started) * 1000)
        return call
    return decorate


class LocalHFRoles:
    """Real HF generation for every app specialist's full Pydantic JSON record.

    ``backend`` may be the same loaded HuggingFaceBackend used by a log-probability
    chooser.  Its generation budget must match ``max_new_tokens``; sharing never
    silently changes another run's declared generation configuration.
    """

    is_llm = True

    def __init__(self, spec, *, backend=None, timeout_seconds=120, max_new_tokens=4096):
        require_slurm()
        self.timeout_seconds = _positive(timeout_seconds, "timeout_seconds")
        self.max_tokens = _integer(max_new_tokens, "max_new_tokens")
        self.spec = dict(spec)
        if self.spec.get('context_extension') not in (None, 'qwen25-yarn-v1'):
            raise ValueError('Unknown explicit expert context extension')
        if self.spec.get("kind", self.spec.get("backend", "hf")) not in {"hf", "huggingface"}:
            raise ValueError("LocalHFRoles requires an offline Hugging Face generative model")
        self.spec = {"kind": "hf", "method": "logprobs", **self.spec}
        if "max_new_tokens" in self.spec and self.spec["max_new_tokens"] != self.max_tokens:
            raise ValueError("Expert spec max_new_tokens differs from its declared JSON budget")
        self.spec["max_new_tokens"] = self.max_tokens
        self.backend = backend if backend is not None else HuggingFaceBackend(self.spec)
        _check_shared_identity(self.spec, self.backend)
        if self.backend.max_new_tokens != self.max_tokens:
            raise ValueError("Shared expert backend must have the declared JSON generation budget")
        self._model_lock = _lock(self.backend)
        self.model_id = str(self.spec.get("model_id") or self.spec.get("model_path") or "local-hf")
        self.backend_id = "local-hf-specialists:" + self.model_id
        self.deadline = float("inf")
        self.last_call = None
        self.structured_output = self.spec.get('structured_output')
        if self.structured_output not in (None, 'json_schema_lmfe_v4'):
            raise ValueError('Unknown explicit expert structured-output protocol')
        self.record_protocol = self.spec.get('record_protocol')
        if self.record_protocol not in (None, 'menu-ids-v1', 'menu-ids-v2', 'menu-ids-v3'):
            raise ValueError('Unknown explicit expert record protocol')
        self._json_guidance = None

    def manifest(self):
        return {"backend_id": self.backend_id, "backend": "offline_hf_specialists",
                "record_protocol": self.record_protocol,
                **({'context_extension': self.spec['context_extension']}
                   if self.spec.get('context_extension') is not None else {}),
                **({'specialist_input_layout': 'literal readable current-state/menu role view; original payload retained in graph trace'}
                   if self.record_protocol == 'menu-ids-v2' else {}),
                **({'specialist_input_layout': 'readable current-state role view with exact named duplicate references and literal grouped menus; original payload retained in graph trace'}
                   if self.record_protocol == 'menu-ids-v3' else {}),
                "procedure_transport": "exact selected-action envelope, no inference" if self.record_protocol else "model generated",
                "verification_citation_policy": (
                    "current public observation-wrapper IDs; empty citations permitted; no-observation wire field omitted and bound []"
                    if self.record_protocol else "original application schema"),
                "structured_output": self.structured_output,
                "structured_output_identity": deepcopy(self._json_guidance.identity) if self._json_guidance else None,
                "model": deepcopy(self.backend.identity), "temperature": 0,
                "max_tokens": self.max_tokens, "timeout_seconds": self.timeout_seconds,
                "transport_retries": 0, "output_mode": "strict JSON object + app Pydantic schema",
                "context_policy": "error_without_truncation",
                "deadline_policy": "HF max_time checked each decode iteration; post-call deadline rejection",
                "probability_semantics": "none; structured specialist generation",
                "source": "comparison.HuggingFaceBackend._generate + original app schemas"}

    @_fresh_call('specialist')
    def invoke(self, role, prompt, payload, schema):
        require_slurm()
        original_schema, original_payload = schema, payload
        menu_request = None
        if self.record_protocol:
            from graph_local.menu_records import prepare_menu_request, procedure_envelope
            if role == 'procedure':
                start = time.monotonic()
                result = procedure_envelope(payload, schema)
                self.last_call = {"backend_id": self.backend_id, "model_id": self.model_id,
                    "role": role, "request_sent": False, "usage": None,
                    "record_protocol": self.record_protocol,
                    "response_classification": "deterministic_selected_action_envelope",
                    "source_payload_sha256": _hash(payload),
                    "elapsed_ms": (time.monotonic() - start) * 1000}
                return result
            menu_request = prepare_menu_request(role, payload, schema, protocol=self.record_protocol)
            prompt, payload, schema = menu_request.prompt, menu_request.payload, menu_request.schema
        schema_json = schema.model_json_schema()
        system = (prompt + "\nReturn one JSON object matching this schema exactly. "
                  "No Markdown, reasoning prose, or extra fields.\n" + _json(schema_json))
        if self.record_protocol in ('menu-ids-v2', 'menu-ids-v3'):
            from graph_local.specialist_context import readable_payload_json
            payload_text = readable_payload_json(payload)
        else:
            payload_text = _json(payload)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": payload_text}]
        start = time.monotonic()
        self.last_call = {"backend_id": self.backend_id, "model_id": self.model_id,
                          "role": role, "request_sent": False, "response": None, "usage": None,
                          "request": {"messages": deepcopy(messages), "max_tokens": self.max_tokens,
                                      "temperature": 0, "stream": False},
                          "prompt_hash": hashlib.sha256(system.encode()).hexdigest(),
                          "schema_hash": _hash(schema_json), "context_policy": "error_without_truncation"}
        if menu_request is not None:
            self.last_call.update(record_protocol=self.record_protocol,
                source_payload_sha256=_hash(original_payload),
                menu_binding=deepcopy(menu_request.audit))
        try:
            timeout = _remaining(self.deadline, self.timeout_seconds)
            call_deadline = start + timeout
            with self._model_lock:
                _remaining(min(self.deadline, call_deadline), self.timeout_seconds)
                prefix = None
                if self.structured_output is not None:
                    from graph_local.guided_json import JsonSchemaGuidance
                    if self._json_guidance is None:
                        # This new wire protocol bounds consecutive whitespace;
                        # the historical structural-v4 default remains unchanged.
                        guidance_options = ({'max_consecutive_whitespaces': 16}
                            if self.record_protocol in ('menu-ids-v1', 'menu-ids-v2', 'menu-ids-v3') else {})
                        self._json_guidance = JsonSchemaGuidance(self.backend.tokenizer,
                            **guidance_options)
                    prefix = self._json_guidance.for_schema(schema_json)
                    self.last_call['structured_output'] = prefix.receipt()
                    _remaining(min(self.deadline, call_deadline), self.timeout_seconds)
                # max_time is a documented GenerationConfig setting consumed by
                # the unchanged HF generate call.  Prefill or a single decoding
                # iteration can overrun it; such output is rejected below.
                config = self.backend.model.generation_config
                previous = getattr(config, "max_time", None)
                config.max_time = max(1e-6, min(call_deadline, self.deadline) - time.monotonic())
                self.last_call["generation_max_time_s"] = config.max_time
                try:
                    self.last_call["request_sent"] = True
                    if prefix is None:
                        raw, input_tokens, output_tokens = self.backend._generate(messages, sample=False, seed=0)
                    else:
                        raw, input_tokens, output_tokens = self.backend._generate(messages, sample=False, seed=0,
                            prefix_allowed_tokens_fn=prefix)
                        self.last_call['structured_output'] = prefix.receipt()
                finally:
                    config.max_time = previous
            self.last_call.update(
                usage=_usage(input_tokens, output_tokens),
                generation=deepcopy(getattr(self.backend, "last_generation_receipt", {})),
                response={"choices": [{"message": {"content": raw},
                                       "finish_reason": "length" if output_tokens >= self.max_tokens else "stop"}]})
            if time.monotonic() >= min(call_deadline, self.deadline):
                self.last_call["response_classification"] = "expert_deadline_exceeded"
                raise TimeoutError("Local specialist exceeded its call or episode deadline; completion rejected")
            if output_tokens >= self.max_tokens:
                self.last_call["response_classification"] = "expert_output_truncated"
                raise ExpertResponseError("expert_output_truncated", "Local specialist reached its output-token limit")
            if not isinstance(raw, str) or not raw.strip():
                self.last_call["response_classification"] = "expert_empty_output"
                raise ExpertResponseError("expert_empty_output", "Local specialist produced no text completion")
            parsed = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if not isinstance(parsed, dict):
                raise ValueError("Specialist must return one JSON object")
            # JSON numeric overflow (for example 1e999) can become infinity even
            # without a NaN/Infinity token, including inside dict[str, Any].
            _json(parsed)
            result = schema.model_validate(parsed).model_dump(mode="json")
            if menu_request is not None:
                result = menu_request.bind(result)
                result = original_schema.model_validate(result).model_dump(mode='json')
                self.last_call['bound_application_record'] = deepcopy(result)
            self.last_call["response_classification"] = "valid_schema_json"
            return result
        except Exception as error:
            self.last_call["error"] = f"{type(error).__name__}: {error}"
            self.last_call.setdefault("response_classification", "expert_schema_error" if isinstance(error, ValueError)
                                      else "expert_infrastructure_error")
            raise
        finally:
            self.last_call["elapsed_ms"] = (time.monotonic() - start) * 1000


class LocalDecisionBackend:
    """One offline chooser used at actual app router and operation boundaries.

    HF uses genuine conditional canonical-label next-token probabilities.  Native
    OpenJev/APUS uses its released learned decision head.  Frozen feature artifacts
    can be transferred without fitting on graph tasks.  Every successful call must
    return a complete vector over the exact offered menu and its legal argmax.
    """

    allow_rules_fallback = False

    def __init__(self, spec, *, backend=None, timeout_seconds=120):
        require_slurm()
        self.spec = dict(spec)
        self.kind = self.spec.get("kind", self.spec.get("backend"))
        if self.kind not in {"hf", "huggingface", "native", "features"}:
            raise ValueError("Graph chooser supports local HF logprobs, native heads or frozen features")
        self.context_protocol = self.spec.get('context_protocol')
        if self.context_protocol not in (None, 'decision-view-v1', 'decision-view-v2', 'decision-view-v3', 'decision-view-v4'):
            raise ValueError('Unknown explicit decision context protocol')
        self.context_max_bytes = _integer(self.spec.get('context_max_bytes', 16000), 'context_max_bytes')
        self.method = {"hf": "logprobs", "huggingface": "logprobs", "native": "native",
                       "features": "features"}[self.kind]
        if self.kind in {"hf", "huggingface"} and self.spec.get("method", "logprobs") != "logprobs":
            raise ValueError("Graph HF chooser requires actual complete logprobs, not generated probabilities")
        if self.kind == "native" and self.spec.get("method", "native") != "native":
            raise ValueError("Graph native chooser requires its released native head")
        if self.kind != "features":
            self.spec["method"] = self.method
        self.timeout_seconds = _positive(timeout_seconds, "timeout_seconds")
        self.backend = backend if backend is not None else build_backend(self.spec)
        _check_shared_identity(self.spec, self.backend)
        self._model_lock = _lock(self.backend)
        self.is_llm = self.kind in {"hf", "huggingface"}
        self.is_decision_model = not self.is_llm
        identity = self.spec.get("model_id") or self.spec.get("model_path") or self.spec.get("artifact_path")
        self.backend_id = "local-graph-chooser:" + self.kind + ":" + str(identity)
        if self.kind == "native":
            self.backend_id += ":" + str(self.spec.get("family", "openjev")) + ":" + str(self.spec.get("effort", "high"))
        self.deadline = float("inf")
        self.last_call = None

    def manifest(self):
        semantics = {"hf": "next_token_softmax_conditional_on_offered_canonical_labels",
                     "huggingface": "next_token_softmax_conditional_on_offered_canonical_labels",
                     "native": "released native learned-head distribution over offered candidates",
                     "features": "frozen fitted candidate scores; transfer to graph without additional fitting"}
        return {"backend_id": self.backend_id, "model": deepcopy(self.backend.identity),
                "context_protocol": self.context_protocol,
                "context_max_bytes": self.context_max_bytes if self.context_protocol else None,
                **({'native_option_placement': 'question_criteria_only'}
                   if self.context_protocol in ('decision-view-v2', 'decision-view-v3', 'decision-view-v4') and self.kind == 'native' else {}),
                **({'model_context_policy': 'decision content; projection hashes and bookkeeping retained in trace'}
                   if self.context_protocol == 'decision-view-v3' else {}),
                **({'model_context_policy': 'exact public evidence and candidate criteria; parameter schemas retained in full trace and native validators'}
                   if self.context_protocol == 'decision-view-v4' else {}),
                "readout": self.method, "probability_semantics": semantics[self.kind],
                "calibration_on_graph_tasks_established": False,
                "context_policy": "error_without_truncation", "rules_fallback": False,
                "timeout_seconds": self.timeout_seconds,
                "deadline_policy": "reject before/after inference; native or forward call is not preempted"}

    def _select(self, purpose, messages, label_to_id, descriptions, *, public_input):
        require_slurm()
        start = time.monotonic()
        self.last_call = {"backend_id": self.backend_id, "purpose": purpose,
                          "request_sent": False, "usage": None,
                          "public_input": deepcopy(public_input), "messages": deepcopy(messages),
                          "options": dict(label_to_id), "option_descriptions": deepcopy(descriptions),
                          "probability_semantics": self.manifest()["probability_semantics"],
                          "context_policy": "error_without_truncation"}
        try:
            timeout = _remaining(self.deadline, self.timeout_seconds)
            call_deadline = start + timeout
            with self._model_lock:
                _remaining(min(self.deadline, call_deadline), self.timeout_seconds)
                self.last_call["request_sent"] = True
                if hasattr(self.backend, "deadline"):
                    self.backend.deadline = min(self.deadline, call_deadline)
                result = self.backend.decide(messages, list(label_to_id),
                                             option_descriptions=descriptions, method=self.method)
            self.last_call["response"] = deepcopy(result)
            self.last_call["usage"] = _usage(result.get("input_tokens"), result.get("output_tokens"))
            self.last_call["backend_identity"] = deepcopy(self.backend.identity)
            if time.monotonic() >= min(call_deadline, self.deadline):
                raise TimeoutError("Local chooser exceeded its call or episode deadline; decision rejected")
            if result.get("status") != "ok":
                raise RuntimeError("Local chooser inference failed: " + str(result.get("error", "unknown backend failure")))
            probabilities = result.get("probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != set(label_to_id):
                raise ValueError("Chooser must supply probabilities for every offered canonical label")
            if any(isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p)
                   or not 0 <= p <= 1 for p in probabilities.values()):
                raise ValueError("Chooser returned invalid probabilities")
            if not math.isclose(sum(probabilities.values()), 1., rel_tol=0, abs_tol=1e-5):
                raise ValueError("Chooser probabilities must sum to one")
            choice = result.get("selected_option_id")
            if choice not in label_to_id or probabilities[choice] != max(probabilities.values()):
                raise ValueError("Chooser must select a legal maximum-probability option")
            return label_to_id[choice], {label_to_id[label]: float(probabilities[label]) for label in label_to_id}
        except Exception as error:
            self.last_call["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.last_call["elapsed_ms"] = (time.monotonic() - start) * 1000

    @_fresh_call('routing')
    def decide(self, summary, allowed_routes):
        wire_summary = deepcopy(summary)
        if self.context_protocol:
            from graph_local.decision_context import build_decision_context
            wire_summary = build_decision_context(summary, purpose='routing',
                allowed_routes=allowed_routes, max_bytes=self.context_max_bytes,
                exact_observation_refs=self.context_protocol == 'decision-view-v4')
            # v2 changes only option placement for native heads. The protocol
            # names have equal UTF-8 length, so the checked byte cap is intact.
            wire_summary['protocol'] = self.context_protocol
        recorded_view = deepcopy(wire_summary)
        if self.context_protocol == 'decision-view-v3':
            from graph_local.decision_context import decision_model_content
            wire_summary = decision_model_content(wire_summary)
        elif self.context_protocol == 'decision-view-v4':
            from graph_local.decision_context import decision_model_content_v4
            wire_summary, projection_audit = decision_model_content_v4(wire_summary)
            recorded_view['audit']['model_projection'] = projection_audit
        text, options = option_prompt(wire_summary, list(allowed_routes))
        descriptions = {label: f"{route}: " + text.rsplit(f"\n[{label}] {route}: ", 1)[1].split("\n", 1)[0]
                        for label, route in options.items()}
        if self.context_protocol:
            criteria = {
                'diagnose': 'Propose the next operation when the goal still needs configuration, execution, waiting, investigation or correction and no current selected intent is ready.',
                'prepare_action': 'Send the already selected current intent to the gate. Prefer this when selected_intent is present and no new error invalidates it; do not repeat diagnosis.',
                'verify': 'Assess FINAL goal completion only when public native evidence supports the goal and no selected intent is awaiting execution. An intermediate read or accepted command alone is not final completion.',
                'deliberate': 'Investigate an ambiguity in the supplied current public evidence.',
            }
            descriptions = {label: route + ': ' + criteria[route] for label, route in options.items()}
            text = ('Choose one offered workflow route using state.goal, state.current_observation, '
                'state.selected_intent and state.last_error. The graph stage label is a routing hint, '
                'not proof that the final physical goal holds.\nState:\n' + _json(wire_summary)
                + '\nOptions:\n' + '\n'.join('[' + label + '] ' + descriptions[label] for label in options)
                + '\nAnswer with the letter of the best option only.')
            if self.context_protocol in ('decision-view-v2', 'decision-view-v3', 'decision-view-v4') and self.kind == 'native':
                text = ('Choose one offered workflow route using state.goal, state.current_observation, '
                    'state.selected_intent and state.last_error. The graph stage label is a routing hint, '
                    'not proof that the final physical goal holds. Exact route choices and their '
                    'instructions are supplied in the native question criteria.\nState:\n' + _json(wire_summary))
        if self.context_protocol == 'decision-view-v4':
            text = ('Input/output schemas are omitted; exact candidate arguments, native observations and validation are unchanged. '
                    'proposed_action_ref names the same receipt\'s identical executed_action; '
                    '$decision_ref names the exact cached observations.\n' + text)
        choice, probabilities = self._select(
            "routing", [{"role": "user", "content": text}], options, descriptions,
            public_input={"summary": summary, "allowed_routes": allowed_routes,
                          **({'decision_view': recorded_view} if self.context_protocol else {}),
                          **({'model_decision_view': wire_summary}
                             if self.context_protocol in ('decision-view-v3', 'decision-view-v4') else {})})
        return RouteDecision(backend_id=self.backend_id, choice=choice, probabilities=probabilities,
                             latency_ms=self.last_call["elapsed_ms"], fallback_used=False)

    @_fresh_call('operational_action_selection')
    def choose(self, context, candidates):
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
            raise ValueError("Operational decision supports exactly 1-4 candidate operations")
        values = [ActionCandidate.model_validate(candidate).model_dump(mode="json") for candidate in candidates]
        ids = [candidate["candidate_id"] for candidate in values]
        if len(ids) != len(set(ids)):
            raise ValueError("Operational candidate IDs must be distinct")
        options = {chr(65 + index): candidate["candidate_id"] for index, candidate in enumerate(values)}
        descriptions = {letter: _json(candidate) for letter, candidate in zip(options, values)}
        wire_context = deepcopy(context)
        if self.context_protocol:
            from graph_local.decision_context import build_decision_context
            wire_context = build_decision_context(context, purpose='operational_action_selection',
                candidates=values, max_bytes=self.context_max_bytes,
                exact_observation_refs=self.context_protocol == 'decision-view-v4')
            wire_context['protocol'] = self.context_protocol
            # Candidates are printed once in the explicit option list below.
            wire_context.pop('candidates', None)
        recorded_view = deepcopy(wire_context)
        if self.context_protocol == 'decision-view-v3':
            from graph_local.decision_context import decision_model_content
            wire_context = decision_model_content(wire_context)
        elif self.context_protocol == 'decision-view-v4':
            from graph_local.decision_context import decision_model_content_v4
            wire_context, projection_audit = decision_model_content_v4(wire_context)
            recorded_view['audit']['model_projection'] = projection_audit
        text = (DECISION_PROMPT + "\nPublic context:\n" + _json(wire_context)
                + "\nCandidate operations:\n" + "\n".join(f"[{label}] {descriptions[label]}" for label in options)
                + "\nAnswer with the letter of the best candidate only.")
        if self.context_protocol in ('decision-view-v2', 'decision-view-v3', 'decision-view-v4') and self.kind == 'native':
            # Native backends already put these unchanged descriptions in their
            # question criteria. Keep complete current state in the state branch
            # and each exact candidate object in its option branch only once.
            text = (DECISION_PROMPT + '\nExact candidate objects, identifiers and actions are supplied '
                    'in the native question criteria.\nPublic context:\n' + _json(wire_context))
        if self.context_protocol == 'decision-view-v4':
            text = ('Input/output schemas are omitted; exact candidate arguments, native observations and validation are unchanged. '
                    'proposed_action_ref names the same receipt\'s identical executed_action; '
                    '$decision_ref names the exact cached observations.\n' + text)
        choice, probabilities = self._select(
            "operational_action_selection", [{"role": "user", "content": text}], options, descriptions,
            public_input={"context": context, "candidates": values,
                          **({'decision_view': recorded_view} if self.context_protocol else {}),
                          **({'model_decision_view': wire_context}
                             if self.context_protocol in ('decision-view-v3', 'decision-view-v4') else {})})
        return validate_decision(OperationDecision(backend_id=self.backend_id, choice=choice,
            probabilities=probabilities, latency_ms=self.last_call["elapsed_ms"]), values)


# Explicit aliases make experiment specifications readable without creating a
# second implementation or substituting another readout behind a model name.
LocalHFDecisionBackend = LocalDecisionBackend
LocalNativeDecisionBackend = LocalDecisionBackend
