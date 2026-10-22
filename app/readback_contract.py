"""Match readback field lists using the native benchmark's order-insensitive semantics."""
from copy import deepcopy

from app.models import ActionProposal, HypothesisSet, OperationOptions


def normalize_arguments(capability_id, arguments, capabilities):
    """Select an equivalent declared preset; never invent a new field combination."""
    if capability_id != "workflow.readback":
        return arguments
    fields = arguments.get("state_ids")
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        return arguments
    schema = capabilities.get(capability_id, {}).get("input_schema", {})
    canonical = dict(arguments, state_ids=sorted(fields))
    for preset in schema.get("enum", []):
        if not isinstance(preset, dict):
            continue
        offered = preset.get("state_ids")
        if isinstance(offered, list) and all(isinstance(field, str) for field in offered):
            if canonical == dict(preset, state_ids=sorted(offered)):
                return deepcopy(preset)
    return arguments


def normalize_record(record, capabilities):
    """Keep candidate, selected-intent and proposal representations consistent."""
    def intent(value):
        return value.model_copy(update={"arguments": normalize_arguments(
            value.capability_id, value.arguments, capabilities)})
    if isinstance(record, OperationOptions):
        return record.model_copy(update={"candidates": [candidate.model_copy(
            update={"intent": intent(candidate.intent)}) for candidate in record.candidates]})
    if isinstance(record, HypothesisSet) and record.intent is not None:
        return record.model_copy(update={"intent": intent(record.intent)})
    if isinstance(record, ActionProposal):
        return intent(record)
    return record
