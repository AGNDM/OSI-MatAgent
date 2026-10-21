"""Offline registry-wide native clock integration audit; no model or solver scripts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from app.device_registry import load_registry, open_native, public_capabilities
from app.single_instrument import InstrumentLab, InstrumentTask
from app.models import ActionProposal


def audit(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True)
    rows = []
    for spec in load_registry()['devices']:
        row = {'device': spec['slug']}
        try:
            host, session, instance, description = open_native(spec)
            fields = {k: v for k, v in description['state_slots'].items()
                      if v.get('visibility', 'agent') == 'agent' and not k.startswith(('truth.', 'debug.', 'host.', 'plant.'))}
            sid = next(iter(fields))
            task = InstrumentTask(task_id='native-clock-integration', device=spec['slug'],
                objective='Audit native clock advancement and fresh readback; not a scientific task.',
                predicates=[{'state_id': sid, 'unit': fields[sid]['unit'], 'value': None}])
            lab = InstrumentLab(task, native=(host, session, instance, description))
            row.update(lifecycle=lab.lifecycle.manifest(), native_capabilities=len(public_capabilities(description)),
                       exposed_native_capabilities=sum('::' in k for k in lab.capabilities()),
                       initial_clock=lab._initial['clock'], proposals=[])
            assert row['native_capabilities'] == row['exposed_native_capabilities']
            if not lab.lifecycle.can_step:
                row.update(status='native_clock_api_absent', limitation='No native step/host_step/host_advance/host_set_clock entry point; public device tools remain exposed')
            else:
                for name, args in [('workflow.inspect', {}), ('workflow.await', {'duration_s': 10}), ('workflow.await', {'duration_s': 5})]:
                    p = ActionProposal(sample_id=lab.sample_id, expected_revision=lab.revision,
                         capability_id=name, arguments=args, request_id=str(uuid4()), postcondition='Fresh native clock readback')
                    event = lab.execute(p)
                    row['proposals'].append(event)
                    if lab.termination_reason:
                        raise RuntimeError(lab.termination_reason + ': ' + str(event.get('result'))[:500])
                before_replay = lab.lifecycle.elapsed
                lab.execute(p)
                assert lab.lifecycle.elapsed == before_replay == 15
                native = lab._measurement['native']
                assert abs(native['clock']['time_s'] - row['initial_clock']['time_s'] - 15) < 1e-6
                assert lab._measurement['revision'] == lab.revision
                row.update(status='passed', final_clock=native['clock'], native_io_counts=lab.lifecycle.calls,
                           exact_replay_advanced_time=False)
        except Exception as exc:
            row.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)[:1000]})
        rows.append(row)
    sources = ['app/native_lifecycle.py', 'app/single_instrument.py', 'app/device_catalog.py',
               'app/device_registry.py', 'bench/device_lifecycle.py', 'cases/devices/registry.json',
               'vendor/chemlab/expansion-data-manifest.json']
    payload = {'version': 'native-lifecycle-all-devices-v1', 'api_requests': 0,
        'scope': 'Native clock and public-tool adapter integration, not scientific acquisition acceptance',
        'source_hashes': {name: sha256(Path(name).read_bytes()).hexdigest() for name in sources},
        'total': len(rows), 'passed': sum(r['status'] == 'passed' for r in rows),
        'native_clock_api_absent': sum(r['status'] == 'native_clock_api_absent' for r in rows),
        'failed': sum(r['status'] == 'failed' for r in rows), 'devices': rows}
    (output/'audit.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: payload[k] for k in ['total', 'passed', 'native_clock_api_absent', 'failed']}))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    audit(parser.parse_args().output)
