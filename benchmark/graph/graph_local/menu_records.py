"""Public menu IDs bind model proposals to exact, currently offered actions.

This is an explicit new transport protocol. It supplies no evaluator state or
reference answer. The expert still proposes candidates, the chooser selects,
and the unchanged app gate and native evaluator execute and score the result.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, create_model

from app.models import Record


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def public_evidence_ids(payload):
    """Return only current public observation-wrapper IDs, never packing refs."""
    return list(dict.fromkeys(
        observation['id'] for key in ('inspection', 'measurement')
        if isinstance(observation := payload.get('observations', {}).get(key), dict)
        and isinstance(observation.get('id'), str)))


def procedure_envelope(payload, schema):
    """Copy the chosen intent into the gate's transport fields without inference."""
    intent = payload['intent']
    return schema.model_validate({
        'sample_id': payload['sample_id'], 'expected_revision': payload['expected_revision'],
        'request_id': payload['request_id'], 'capability_id': intent['capability_id'],
        'arguments': deepcopy(intent['arguments']), 'postcondition': intent['postcondition'],
    }).model_dump(mode='json')


@dataclass
class MenuRequest:
    prompt: str
    payload: dict
    schema: type
    audit: dict
    actions: dict | None = None
    omit_verification_evidence_ids: bool = False

    def bind(self, value):
        if self.actions is None:
            if self.omit_verification_evidence_ids:
                if 'evidence_ids' in value:
                    raise ValueError('No-observation verification wire records must omit evidence_ids')
                return {**deepcopy(value), 'evidence_ids': []}
            return value
        candidates = []
        ids = []
        for selected in value['candidates']:
            menu_id = selected['menu_action_id']
            if menu_id not in self.actions or menu_id in ids:
                raise ValueError('Candidates must select distinct currently offered menu IDs')
            ids.append(menu_id)
            action = deepcopy(self.actions[menu_id])
            candidates.append({
                'candidate_id': menu_id,
                'intent': {'capability_id': 'benchmark.' + action['kind'],
                    'arguments': {'action': action},
                    'discriminating_observation': selected['rationale'],
                    'postcondition': selected['postcondition']},
                'rationale': selected['rationale'], 'evidence_ids': selected.get('evidence_ids', []),
                'material_evidence_ids': [],
            })
        evidence = list(dict.fromkeys(item for candidate in candidates for item in candidate['evidence_ids']))
        return {'hypotheses': [{'cause': value['assessment'], 'supporting_ids': evidence,
                               'contradicting_ids': [], 'material_evidence_ids': []}],
                'candidates': candidates}


def prepare_menu_request(role, payload, original_schema, *, protocol='menu-ids-v1'):
    from graph_local.public_context import compact_json

    if protocol not in ('menu-ids-v1', 'menu-ids-v2', 'menu-ids-v3'):
        raise ValueError('Unknown public menu record protocol')
    readable = protocol in ('menu-ids-v2', 'menu-ids-v3')
    source = deepcopy(payload)
    public = source.get('benchmark_public_context')
    if not isinstance(public, dict):
        raise ValueError('menu-ids-v1 requires the public benchmark context')
    evidence_ids = public_evidence_ids(source)
    audit = {'protocol': protocol, 'source_payload_sha256': _hash(payload),
             'public_wrapper_revision': public.get('wrapper_revision'),
             'evidence_ids': evidence_ids,
             'uses_reference_or_evaluator_state': False}
    actions = None
    omit_verification_evidence_ids = False
    schema = original_schema
    if role == 'diagnostic':
        actions, menu = {}, []
        seen = set()
        commands = public.pop('complete_offered_commands_and_presets')
        for command in commands:
            for preset in command['argument_presets']:
                action = preset['action']
                signature = _json(action)
                if signature in seen:
                    continue
                seen.add(signature)
                menu_id = 'm' + str(len(actions))
                actions[menu_id] = deepcopy(action)
                menu.append({'menu_action_id': menu_id, 'action': deepcopy(action),
                             'preset_text': deepcopy(preset.get('text', command['command'].get('text')))}
                            if readable else {'menu_action_id': menu_id,
                             'command': deepcopy(command['command']), 'preset': deepcopy(preset)})
        if not actions:
            raise ValueError('No public actions offered; no candidate may be fabricated')
        public['offered_menu_actions'] = menu
        menu_id_type = Literal.__getitem__(tuple(actions))
        candidate_fields = dict(
            menu_action_id=(menu_id_type, ...),
            rationale=(str, Field(min_length=1, max_length=120 if readable else 240)),
            postcondition=(str, Field(min_length=1, max_length=120 if readable else 240)))
        if evidence_ids:
            citation_type = Literal.__getitem__(tuple(evidence_ids))
            candidate_fields['evidence_ids'] = (list[citation_type], Field(max_length=2))
        # With no observations there is nothing to cite. Omit the wire field and
        # bind [] explicitly; LMFE 0.11.3 admits one item for maxItems:0.
        candidate = create_model('PublicMenuCandidate', __base__=Record, **candidate_fields)
        schema = create_model('PublicMenuCandidates', __base__=Record,
            assessment=(str, Field(min_length=1, max_length=160 if readable else 240)),
            candidates=(list[candidate], Field(min_length=1, max_length=4)))
        prompt = (
            'Choose useful NEXT operations for the stated public goal using current native observations, '
            'manuals, exact native feedback and offered_menu_actions. Each menu entry binds a '
            'menu_action_id to one complete legal action and its exact preset. Propose one to four '
            'distinct menu_action_id values, with short rationales and expected postconditions. '
            'A separate decision model chooses among your candidates. Do not execute or reconstruct '
            'an action, invent parameters, copy an old menu ID from history, or propose actions for '
            'a future state. When no instrument is selected, consider the offered selection actions. '
            'Command acceptance is not physical completion. An offered wait advances simulation; '
            'a read does not. Finish only when the public evidence supports completion; use escalation '
            'only for justified maintenance. Cite only the supplied current inspection/measurement IDs. '
            'The native evaluator alone scores success. Keep every prose field brief.'
        )
        audit.update(menu_action_count=len(actions), menu_actions_sha256=_hash(actions),
                     menu_binding=deepcopy(actions), output_semantics='model-proposed public-menu IDs')
    elif role == 'verification':
        # Keep the application's status values and record shape. Public citation
        # membership and explanation length are constrained only on this opt-in
        # wire schema; the transport supplies no verdict, freshness or outcome.
        verification_fields = {
            name: (field.annotation, deepcopy(field))
            for name, field in original_schema.model_fields.items()
            if name != 'evidence_ids'
        }
        # Bound explanation length only on this explicit wire protocol; the
        # original application schema, verifier verdict and grading stay intact.
        gap_characters = 160 if readable else 240
        verification_fields['missing_evidence'] = (
            list[Annotated[str, Field(max_length=gap_characters)]], Field(max_length=4))
        if evidence_ids:
            citation_type = Literal.__getitem__(tuple(evidence_ids))
            verification_fields['evidence_ids'] = (
                list[citation_type], Field(max_length=min(2, len(evidence_ids))))
        else:
            # LMFE 0.11.3 does not reliably constrain maxItems:0. Leave this
            # field off the wire, then restore the only legal value in bind().
            omit_verification_evidence_ids = True
        schema = create_model('PublicMenuVerification', __base__=Record, **verification_fields)
        audit.update(output_semantics='model-generated verification with current public-wrapper citations',
                     citation_policy='current inspection/measurement wrapper IDs only',
                     missing_evidence_limits={'max_items': 4, 'max_characters_per_item': gap_characters},
                     wire_evidence_ids='omitted and bound to []' if omit_verification_evidence_ids else 'public-ID enum; empty list permitted')
        prompt = (
            'Assess the final public goal using current native readings, their units, quality and '
            'timestamps, and actual executed history. Return supported only for a completion claim '
            'established by the evidence; contradicted for an observed mismatch; unresolved for '
            'missing decisive evidence. A command receipt alone proves no final outcome. Cite only '
            'the current inspection/measurement wrapper IDs supplied in observations. Wrapper revision '
            'authenticates transport and does not make old native readings fresh. The independent '
            'native evaluator assigns the official outcome. List only missing decisive evidence: '
            f'at most four brief items, each at most {gap_characters} characters. State each evidence gap once; '
            'do not repeat sentences or add general discussion. Use an empty list when no evidence is missing.'
        )
        prompt += (' Allowed current wrapper IDs: ' + _json(evidence_ids) + '. '
                   + ('Cite these literal IDs only.' if readable else
                      'Shared-reference names are serialization references, never evidence IDs.'))
        if omit_verification_evidence_ids:
            prompt += ' There are no current wrapper IDs: omit evidence_ids; the transport supplies an empty list.'
    else:
        raise ValueError('Unsupported menu record role: ' + str(role))
    if readable:
        from graph_local.specialist_context import (
            readable_specialist_payload, compact_readable_specialist_payload)
        projection = (compact_readable_specialist_payload if protocol == 'menu-ids-v3'
                      else readable_specialist_payload)
        source, projection_audit = projection(payload, role=role,
            menu=menu if role == 'diagnostic' else None)
        audit.update(projection_audit)
        if protocol == 'menu-ids-v3':
            prompt = prompt.replace(
                'manuals, exact native feedback and offered_menu_actions. Each menu entry binds a '
                'menu_action_id to one complete legal action and its exact preset.',
                'manuals, exact native feedback and offered_action_groups. Each choice inherits '
                'its group\'s literal common_action fields (kind/instrument/capability), then supplies '
                'every remaining native field, including exact arguments. Its menu_action_id binds '
                'that complete legal action. All offered actions remain available.')
        prompt += (
            ('\nInput is readable JSON. A receipt_ref names latest_native_receipt or '
             'latest_successful_native_receipt with the same history_index. An observations_ref '
             'names the exact cached_native_observation map or its observations field; quality, '
             'sample times, values and numeric types are identical. These named references '
             'remove identical copies only. observations_from_cached lists state IDs whose entire '
             'reading exactly equals that ID in the current cache; merge those readings with the '
             'literal response.observations entries. Every different reading remains literal. '
             'preconditions_ref names another capability with the entire identical preconditions list. '
             'action_from_menu_id in action counts names the exact native action in that offered menu entry. '
             'proposed_action_ref="executed_action" means the '
             'proposal exactly equals that same receipt\'s literal executed_action. '
             'No bitmasks or encoded tables are used. '
             if protocol == 'menu-ids-v3' else
             '\nInput is literal readable JSON, not a shared-reference or table encoding. ')
            +
            'Read the exact goal, cached_native_observation and native_receipts_in_execution_order. '
            'latest_native_receipt is the actual last acknowledged response; '
            'latest_successful_native_receipt preserves the last successful command before any later rejection. '
            'Cached readings retain their original quality/times and can precede successful commands '
            'or newer rejections. Compare receipt history indices and clocks; never assume the initial '
            'cached reading remains the latest command outcome. Do not rewrite cached values or infer '
            'that a rejected command with no side effects undid an earlier successful command. '
            'Do not restart the plan from the episode initial state. Do not infer '
            'completion from command acceptance. Reassess the next step after each actual receipt. '
            'When evidence is missing or a command is rejected, consider useful offered reads, waits '
            'or other recovery actions using their literal preconditions; do not repeat the same rejected '
            'command without a changed condition. Reads observe; waits advance simulated time. '
            'Choose proposals yourself from all offered IDs; no preferred or correct action is supplied. '
            'Use concise prose: assessment at most 160 characters, rationale/postcondition at most 120. '
            'Finish only when public evidence supports the complete stated goal.'
        )
        if protocol == 'menu-ids-v3':
            prompt = prompt.replace(
                'latest_successful_native_receipt preserves the last successful command before any later rejection.',
                'latest_successful_native_receipt is the last successful acknowledged response, '
                'which may be a read, wait or command; it is historical if a later receipt exists.')
            if role == 'verification':
                prompt = (
                    "Returning status 'supported' immediately ends this episode as a FINISH claim for the ENTIRE public goal. "
                    'It does not mean an observation is valid, a command succeeded, an instrument was selected, '
                    'or a next operation is allowed. '
                    + prompt)
            elif role == 'diagnostic':
                prompt += (
                    ' historical_last_verification_feedback, when supplied, is the exact previous verifier report '
                    'with its original observation ID and evidence signature. Use its missing-evidence critique '
                    'when reassessing the next operation, without treating its citations as fresh evidence.')
            prompt += (
                ' Parameter schemas are omitted from this role view. Every legal offered action and its '
                'exact arguments are supplied in the diagnostic menu; the native validator is unchanged. '
                'Verification assesses the observed goal evidence and does not propose parameters.')
        return MenuRequest(prompt, source, schema, audit, actions, omit_verification_evidence_ids)
    prompt += (
        '\nPublic input uses a lossless shared-reference/table encoding. $public_ref names an '
        'entry in shared. $public_table.paths lists nested column paths; defaults apply to all '
        'rows. Each row is [presence_bitmask,values]: bit i includes paths[i], with cells in '
        'increasing i; an absent bit means missing, not null. $public_object escapes a literal '
        'object as key/value pairs. Resolve these encodings as exact public '
        'values, not summaries or inferred state. All current readings and offered actions are retained.'
    )
    compacted = compact_json(source)
    audit['serialized_payload_sha256'] = _hash(compacted)
    return MenuRequest(prompt, compacted, schema, audit, actions, omit_verification_evidence_ids)
