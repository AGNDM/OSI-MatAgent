"""Pinned schema-guided JSON transport, independent of evaluator content.

Only the declared wire schema is accepted here. A caller may explicitly supply
public-menu identifier enums in a new record protocol. No reference answers or
private evaluator state are used. The application validates bound records.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

MODE = 'json_schema_lmfe_v4'
VERSION = '0.11.3'
WHITESPACE_LIMIT = 1_000_000_000


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def guidance_schema(schema):
    """Remove explicit unrestricted additionalProperties (release compatibility).

    Omitting ``additionalProperties:true`` preserves JSON Schema semantics.
    The released parser mistakes True for a nested schema object. Closed
    records, required fields, types, optionality and legal status values stay.
    No action/identifier enums or constants are added.
    """
    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()
                    if not (key == 'additionalProperties' and item is True)}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return deepcopy(value)
    return visit(schema)


def _whitespace_limit(value):
    if type(value) is not int or value < 1:
        raise ValueError('max_consecutive_whitespaces must be a positive integer')
    return value


def make_parser(schema, alphabet, *, max_consecutive_whitespaces=WHITESPACE_LIMIT):
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
    config = CharacterLevelParserConfig(alphabet=alphabet,
        max_consecutive_whitespaces=_whitespace_limit(max_consecutive_whitespaces),
        force_json_field_order=False, max_json_array_length=0)
    return JsonSchemaParser(guidance_schema(schema), config=config)


class StructuralJsonPrefix:
    def __init__(self, function, descriptor):
        self.function = function
        self.descriptor = descriptor
        self.calls = 0
        self.minimum_allowed_tokens = None
        self.maximum_allowed_tokens = None

    def __call__(self, batch_id, input_ids):
        allowed = self.function(batch_id, input_ids)
        if not allowed:
            raise RuntimeError('Structural JSON mask returned no legal token; no fallback')
        enforcer = self.function.token_enforcer
        state = enforcer.prefix_states[tuple(input_ids.tolist())]
        # The pinned release catches internal parser exceptions and can silently
        # offer EOS. Surface that transport failure rather than accepting it as
        # a model-generated complete JSON document.
        eos_ids = enforcer.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        if type(state.parser).__name__ == 'ForceStopParser' or (
                eos_ids.intersection(allowed) and not state.parser.can_end()):
            raise RuntimeError('Structural JSON parser failed or offered premature EOS; no fallback')
        self.calls += 1
        count = len(allowed)
        self.minimum_allowed_tokens = count if self.minimum_allowed_tokens is None else min(count, self.minimum_allowed_tokens)
        self.maximum_allowed_tokens = count if self.maximum_allowed_tokens is None else max(count, self.maximum_allowed_tokens)
        return allowed

    def receipt(self):
        return {**deepcopy(self.descriptor), 'prefix_callback_calls': self.calls,
            'minimum_allowed_tokens': self.minimum_allowed_tokens,
            'maximum_allowed_tokens': self.maximum_allowed_tokens}


class JsonSchemaGuidance:
    """Cache tokenizer preprocessing, but create a new parser for every call."""
    def __init__(self, tokenizer, *, max_consecutive_whitespaces=WHITESPACE_LIMIT):
        self.max_consecutive_whitespaces = _whitespace_limit(max_consecutive_whitespaces)
        if importlib.metadata.version('lm-format-enforcer') != VERSION:
            raise RuntimeError('Structured decoding requires pinned lm-format-enforcer==' + VERSION)
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        started = time.monotonic()
        self.tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
        self.initialization_wall_time_s = time.monotonic() - started
        self.identity = {'mode': MODE, 'library': 'lm-format-enforcer', 'version': VERSION,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'tokenizer_data_preprocessing_wall_time_s': self.initialization_wall_time_s,
            'tokenizer_alphabet_sha256': hashlib.sha256(self.tokenizer_data.tokenizer_alphabet.encode()).hexdigest(),
            'schema_normalization': 'Remove only additionalProperties:true; unrestricted default is equivalent',
            'force_json_field_order': False, 'max_json_array_length': 0,
            'max_consecutive_whitespaces': self.max_consecutive_whitespaces,
            'additional_action_filtering_beyond_supplied_schema': False,
            'oracle_or_payload_access': False,
            'fail_closed_on_library_eos_fallback': True,
            'post_generation_validation': 'Unchanged strict JSON/Pydantic/evidence/preset/application gate'}

    def for_schema(self, schema):
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        started = time.monotonic()
        parser = make_parser(schema, self.tokenizer_data.tokenizer_alphabet,
            max_consecutive_whitespaces=self.max_consecutive_whitespaces)
        requested_config = parser.config
        function = build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, parser)
        # LMFE 0.11.3 replaces the parser's entire config during construction.
        # Restore only this parser instance, preserving its tokenizer alphabet.
        # No environment/global setting or model method is mutated.
        parser.config = requested_config
        assert parser.config.alphabet == self.tokenizer_data.tokenizer_alphabet
        assert parser.config.max_json_array_length == 0
        assert parser.config.force_json_field_order is False
        assert parser.config.max_consecutive_whitespaces == self.max_consecutive_whitespaces
        descriptor = {**self.identity, 'application_schema_sha256': _hash(schema),
            'guidance_schema_sha256': _hash(guidance_schema(schema)),
            'schema_compile_wall_time_s': time.monotonic() - started,
            'parser_configuration_restored_after_release_builder': True}
        return StructuralJsonPrefix(function, descriptor)
