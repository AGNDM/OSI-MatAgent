"""Real LMFE whitespace limits for the opt-in public menu record protocol."""
import json
import os
import string

import pytest

if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
    pytest.skip("Whitespace guidance tests require a genuine Slurm allocation", allow_module_level=True)

from lmformatenforcer.consts import COMPLETE_ALPHABET
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from app.models import VerificationReport
from graph_local.backends import LocalHFRoles
from graph_local.guided_json import JsonSchemaGuidance, WHITESPACE_LIMIT, make_parser
from test_backends import FakeHF


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"operation": {"type": "string"}, "note": {"type": "string"}},
    "required": ["operation", "note"],
}


def _accepts(text, *, maximum=None, schema=SCHEMA):
    kwargs = {} if maximum is None else {"max_consecutive_whitespaces": maximum}
    parser = make_parser(schema, COMPLETE_ALPHABET, **kwargs)
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


@pytest.mark.parametrize("indent", [None, 2, 4])
def test_finite_limit_accepts_compact_and_normal_formatted_json(indent):
    value = {"operation": "an.offered.or.unoffered.identifier", "note": "Use actual public evidence."}
    assert _accepts(json.dumps(value, indent=indent), maximum=16)


def test_finite_limit_accepts_sixteen_layout_spaces_but_blocks_a_seventeenth():
    before = '{"operation":"public.choice",'
    after = '"note":"Short public rationale"}'
    assert _accepts(before + " " * 16 + after, maximum=16)
    assert not _accepts(before + " " * 17 + after, maximum=16)
    assert not _accepts(before + " \t\r\n" * 8 + after, maximum=16)


def test_default_preserves_historical_v4_whitespace_budget():
    parser = make_parser(SCHEMA, COMPLETE_ALPHABET)
    assert parser.config.max_consecutive_whitespaces == WHITESPACE_LIMIT == 1_000_000_000
    assert _accepts('{"operation":"public.choice",' + " \t\n" * 32 + '"note":"still legal in v4"}')


def test_finite_whitespace_budget_does_not_add_action_or_citation_enums():
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {"action": {"type": "string"},
                       "citations": {"type": "array", "items": {"type": "string"}}},
        "required": ["action", "citations"],
    }
    value = {"action": "deliberately.not.an.offered.action", "citations": ["invented-evidence-id"]}
    assert _accepts(json.dumps(value), maximum=16, schema=schema)


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5, "16"])
def test_invalid_whitespace_budgets_fail_explicitly(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        make_parser(SCHEMA, COMPLETE_ALPHABET, max_consecutive_whitespaces=maximum)


def _tokenizer():
    characters = "".join(dict.fromkeys(string.ascii_letters + string.digits + string.punctuation + " \n\t"))
    vocabulary = {"[UNK]": 0, "<eos>": 1}
    vocabulary.update({character: index + 2 for index, character in enumerate(characters)})
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    raw.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="<eos>")


@pytest.mark.parametrize("maximum", [16, WHITESPACE_LIMIT])
def test_actual_token_enforcer_receipt_records_requested_budget(maximum):
    guidance = JsonSchemaGuidance(_tokenizer(), max_consecutive_whitespaces=maximum)
    prefix = guidance.for_schema(SCHEMA)
    assert guidance.identity["max_consecutive_whitespaces"] == maximum
    assert prefix.receipt()["max_consecutive_whitespaces"] == maximum
    assert prefix.receipt()["parser_configuration_restored_after_release_builder"] is True
    assert prefix.receipt()["oracle_or_payload_access"] is False


@pytest.mark.parametrize("record_protocol, expected", [(None, WHITESPACE_LIMIT), ("menu-ids-v1", 16)])
def test_role_backend_uses_finite_budget_only_for_new_menu_protocol(record_protocol, expected):
    raw = {"status": "unresolved", "evidence_ids": ["obs-7"], "missing_evidence": ["A fresh native reading"]}
    backend = FakeHF(raw=json.dumps(raw))
    backend.tokenizer = _tokenizer()
    spec = {"model_path": "/fixture/not-loaded", "structured_output": "json_schema_lmfe_v4"}
    if record_protocol:
        spec["record_protocol"] = record_protocol
    roles = LocalHFRoles(spec, backend=backend)
    payload = {"observations": {"inspection": {"id": "obs-7"}, "measurement": None},
               "benchmark_public_context": {"wrapper_revision": 7, "current_public_view": {},
                                            "complete_native_history": [],
                                            "complete_offered_commands_and_presets": []}}
    result = roles.invoke("verification", "Use public evidence.", payload, VerificationReport)
    assert result["status"] == "unresolved"
    assert roles.last_call["structured_output"]["max_consecutive_whitespaces"] == expected
    assert roles.manifest()["structured_output_identity"]["max_consecutive_whitespaces"] == expected
