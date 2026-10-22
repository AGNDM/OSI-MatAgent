"""Public-only instrument benchmark profile for the unchanged application graph."""
from copy import deepcopy


class BenchmarkProfile:
    prompts = {
        'diagnostic': 'Operate the supplied instrument benchmark using only its public goal, current cached readings, full native history, manuals and offered commands/presets. Propose one to four useful alternatives. Each intent capability_id is benchmark.<native kind>; arguments must be exactly {"action": <one complete offered native action JSON>}. Copy an offered preset without inventing parameters. A command acceptance does not prove completion; reads retain their original times and quality. Use explicit native wait to advance simulation. Diagnose failures from actual feedback and request useful reads before intervention when needed. Finish only with public evidence of the goal; escalate only when public evidence justifies maintenance. Cite observation wrapper IDs from the supplied inspection/measurement, not state-slot names. Never infer evaluator fixtures, grading checkpoints or future faults.',
        'procedure': 'Copy the selected intent into one ActionProposal. Copy sample_id, expected_revision and request_id exactly. Copy capability_id and arguments without changing the embedded native action. Do not execute anything.',
        'verification': 'Independently assess whether the supplied public benchmark goal is complete using the full public evidence and actual native history. Supported is a completion CLAIM, not an official score. Cite the current inspection/measurement evidence ID. Native commands may start continuing processes; verify final state, readback, elapsed time and required evidence. Original reading timestamps and qualities remain decisive; a valid wrapper only authenticates snapshot transport, not fresh physical measurement. Return contradicted for an observed mismatch or unresolved for missing decisive evidence. Do not assign success because an action was accepted or because the diagnostic requested finish.',
        'deliberation': 'Choose one supplied allowed specialist route from the complete public context. Do not execute native commands.'}
    candidate_prompt = (prompts['diagnostic'] +
        ' When no instrument is selected, propose exactly ONE literal offered select_instrument action; the catalog wrapper is not a physical reading.'
        ' At every later step, propose between one and four DISTINCT literal offered native actions; one well-supported candidate is sufficient.'
        ' Candidate IDs must be distinct, and capability/arguments pairs must also differ. Rewording a rationale does not create a different action.'
        ' There is no workflow.inspect alias: the complete public command menu defines the available benchmark.<kind> aliases.')

    def __init__(self, lab, record_protocol=None):
        self.lab = lab
        self.record_protocol = record_protocol

    def projection(self, role, state, request_id):
        from app.roles import projection
        payload = projection(role, state, request_id)
        payload['benchmark_public_context'] = self.lab.context()
        if self.record_protocol == 'menu-ids-v3' and role == 'diagnostic':
            # This is the previous public assessment, not fresh native evidence.
            payload['last_verification_feedback'] = deepcopy(state.get('last_verification_feedback'))
        return payload

    @staticmethod
    def retrieve(role):
        return []  # Native public manuals are already present in full context.

    def validate_role_contract(self, role, record, payload):
        if role == 'diagnostic' and record.intent is not None:
            self.lab.intent_action(record.intent)
        elif role == 'procedure':
            for name in ('sample_id', 'expected_revision', 'request_id'):
                if getattr(record, name) != payload[name]:
                    raise ValueError('Procedure must copy ' + name + ' exactly')
            if record.capability_id != payload['intent']['capability_id'] or record.arguments != payload['intent']['arguments']:
                raise ValueError('Procedure changed the selected native action')
        elif role == 'verification':
            known = {o['id'] for key in ('inspection', 'measurement') if (o := payload['observations'].get(key))}
            if record.status == 'supported' and not (set(record.evidence_ids) & known):
                raise ValueError('Completion claim must cite current public evidence')
            # No evaluator-side goal predicates are evaluated or supplied here.
