"""Public selection handoff for the autonomous router; performs no runtime I/O."""
from copy import deepcopy


POLICY_VERSION = 'autonomous-selected-intent-v1'
ROUTING_INSTRUCTION = (
    'Route the workflow using selected_intent and its selection revision. '
    'When selected_intent_current is true and last_error is absent, choose prepare_action '
    'to prepare that exact action. Sensor snapshot age is separate from action-selection age: '
    'an old snapshot does not invalidate an action just selected for the current revision. '
    'In particular, a selected readback may be intended to refresh that old snapshot; '
    'prepare it instead of planning the same read again. '
    'Choose diagnose when no confirmed current selection exists or a new error requires replanning. '
    'Do not choose observation fields, timing or instrument controls at this routing stage; '
    'those are specified by the selected action. Choose only an offered route.\n'
)
ROUTING_CRITERIA = {
    'diagnose': 'Plan when no confirmed action is selected for the current revision, or a new error requires revising it. Old sensor data alone does not invalidate a current selection.',
    'prepare_action': 'Prepare the exact selected_intent for execution when selected_intent_current is true and last_error is absent. Preserve its capability and arguments.',
}


def selected_intent_context(state, current_revision):
    intent = deepcopy(state.get('intent'))
    decision = state.get('operation_decision')
    selection_revision = state.get('operation_revision') if decision and intent else None
    current = bool(intent and decision and selection_revision == current_revision
                   and state.get('observation_revision') == current_revision)
    return dict(routing_context_policy=POLICY_VERSION,
        selected_intent=intent,
        selected_candidate_id=decision.get('choice') if decision and intent else None,
        selected_intent_revision=selection_revision,
        current_revision=current_revision, selected_intent_current=current,
        last_error=deepcopy(state.get('last_error')))
