"""Pinned native devices. Only host-selected modules/classes may be instantiated."""
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import sys

from app.native import VENDOR_ROOT

REGISTRY = Path(__file__).resolve().parents[1] / 'cases/devices/registry.json'
for path in (VENDOR_ROOT, VENDOR_ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def open_native(spec):
    # Specs are developer-owned registry entries, never LLM-provided entrypoints.
    name = spec['module']
    if not (name.startswith('asset_runtime.') or name.startswith('basic40_runtime')):
        raise ValueError('Unapproved native module')
    module = importlib.import_module(name)
    cls = getattr(module, spec['class'])
    constructor = inspect.signature(cls)
    required = [p for p in constructor.parameters.values() if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    host = cls(instance_id='device-01') if len(required) == 1 and required[0].name == 'instance_id' else cls()
    instance = getattr(host, 'instance_id', None)
    if hasattr(host, 'create_instance'):
        instance = 'device-01'
        create = inspect.signature(host.create_instance)
        if 'asset_type_id' in create.parameters:
            host.create_instance(instance, spec.get('runtime_asset_id', spec['asset_id']))
        else:
            host.create_instance(instance)
    if not instance:
        raise ValueError('No known native instance contract')
    # Explicit developer-owned ambient environment setup, never an agent capability.
    fixture_events = []
    plant = getattr(module, 'CAPS', {}).get('plant.configure')
    if not hasattr(host, 'create_instance') and plant:
        host.register_principal('fixture', visibility='evaluator')
        fixture = host.session('fixture')
        declaration = fixture.describe(instance_id=instance)['capabilities']['plant.configure']
        example = declaration.get('example_arguments')
        if isinstance(example, list):
            example = example[0]
        if not isinstance(example, dict):
            raise ValueError('No declared native environment fixture')
        # Mechanical preconditions from the pinned upstream configured() tests.
        preconditions = {
            'asset_runtime.syrris_asiadiscovery':[('clamp.set_lift', {'lift_m':.04})],
            'asset_runtime.thalesnano_hcube':[('clamp.set_lift', {'lift_m':.05})],
            'asset_runtime.cem_libertyblue':[('cover.set_lift', {'lift_m':.08}), ('clamp.set_lift', {'lift_m':.04})],
            'asset_runtime.cytiva_ktaoligosynt':[('clamp.set_lift', {'lift_m':.04})],
        }.get(name, [])
        steps = preconditions + [('plant.configure', example)]
        steps += [(cap, {key:0. for key in args}) for cap,args in reversed(preconditions)]
        for index, (cap, args) in enumerate(steps):
            setup = fixture.invoke(instance_id=instance, capability_id=cap,
                arguments=deepcopy(args), request_id=f'fixture-{index}')
            fixture_events.append({'capability_id':cap, 'arguments':args, 'response':setup})
            if setup.get('status') != 'succeeded':
                raise ValueError('Native environment setup rejected: ' + str(setup.get('error')))
    host._workflow_fixture_events = fixture_events
    sig = inspect.signature(host.register_principal)
    if 'visibility' in sig.parameters:
        host.register_principal('workflow', visibility='agent')
    else:
        host.register_principal('workflow')
    session = host.session('workflow')
    description = session.describe(instance_id=instance)
    # Some older hosts wrap the selected asset metadata.
    if 'asset_types' in description:
        description = description['asset_types'][spec['asset_id']]
    return host, session, instance, description


def public_capabilities(description):
    result = {}
    for name, cap in description.get('capabilities', {}).items():
        if cap.get('visibility', 'agent') != 'agent':
            continue
        if cap.get('availability', 'implemented') != 'implemented':
            continue
        # Never promote setup or simulation control even if upstream metadata errs.
        if name.startswith(('sim.', 'host.', 'plant.')):
            continue
        if not isinstance(cap.get('input_schema'), dict):
            continue
        normalized = deepcopy(cap)
        if isinstance(normalized.get('example_arguments'), list):
            normalized['example_arguments'] = next((x for x in normalized['example_arguments'] if isinstance(x, dict)), None)
        result[name] = normalized
    return result


def load_registry():
    return json.loads(REGISTRY.read_text())


def get_device(key):
    return next(x for x in load_registry()['devices'] if key in {x['asset_id'], x['slug'], x['chem_id']})


def load_task(key):
    device = get_device(key)
    return device, json.loads((REGISTRY.parent / device['task_file']).read_text())
