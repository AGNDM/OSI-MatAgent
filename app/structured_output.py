"""Bind operation arguments to the public capability schema before generation.

Fireworks accepts JSON Schema 2020-12, including optional properties and anyOf.
Do not insert nulls/defaults, repair responses, or prescribe operation values.
"""
from copy import deepcopy

from jsonschema import Draft202012Validator


def _embedded_schema(source, pointer):
    """Keep native in-document references valid inside the response document."""
    if isinstance(source, list):
        return [_embedded_schema(value, pointer) for value in source]
    if not isinstance(source, dict):
        return source
    result = {}
    for key, value in source.items():
        if key == '$ref':
            if not value.startswith('#'):
                raise ValueError('Strict output requires in-document capability schema references')
            result[key] = pointer + value[1:]
        else:
            result[key] = _embedded_schema(value, pointer)
    return result


def output_schema(model, payload):
    """Retain the workflow envelope and correlate capability_id with arguments."""
    document = deepcopy(model.model_json_schema())
    definitions = document.setdefault('$defs', {})
    intent = definitions.get('Intent')
    proposal = model.__name__ == 'ActionProposal'
    if intent is not None or proposal:
        capabilities = payload.get('capabilities')
        if not capabilities:
            raise ValueError('Strict operation output requires public capability input schemas')
        alternatives = []
        for index, (capability_id, capability) in enumerate(capabilities.items()):
            native = capability.get('input_schema')
            if not isinstance(native, dict):
                raise ValueError('Missing input_schema for ' + capability_id)
            name = f'CapabilityArguments_{index}'
            if name in definitions:
                raise ValueError('Conflicting schema definition: ' + name)
            pointer = '#/$defs/' + name
            definitions[name] = _embedded_schema(native, pointer)
            branch = deepcopy(intent if intent is not None else document)
            branch.pop('$defs', None)
            branch['properties']['capability_id'] = {'type': 'string', 'enum': [capability_id]}
            branch['properties']['arguments'] = {'$ref': pointer}
            alternatives.append(branch)
        if intent is not None:
            definitions['Intent'] = {'anyOf': alternatives}
        elif len(alternatives) == 1:
            document = dict(alternatives[0], **{'$defs': definitions})
        else:
            # Procedure payloads normally contain just the selected capability.
            document = {'anyOf': alternatives, '$defs': definitions}
    if not document.get('$defs'):
        document.pop('$defs', None)
    Draft202012Validator.check_schema(document)
    return document
