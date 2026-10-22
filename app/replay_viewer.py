"""Read-only, instrument-keyframe replay. Never instantiate or advance a simulator."""
from bisect import bisect_right
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from app.result_viewer import (brief, list_runs, metadata, read_events, read_json,
                               read_text, run_directory, sanitize)

MODEL_DIR = Path('vendor/chemlab/assets/expansion_batch04/t1_thermal/mt_rc1mxhfcal')
VENDOR_FILES = ('build/three.module.js', 'build/three.core.js',
                'examples/jsm/loaders/GLTFLoader.js', 'examples/jsm/controls/OrbitControls.js',
                'examples/jsm/utils/BufferGeometryUtils.js')
KINDS = {'tool_result', 'instrument_score', 'agent_report', 'rejected', 'role_error',
         'operation_error', 'bounded_failure'}
READ_KEYS = {'reactor.indicated_k', 'jacket.indicated_k', 'stirrer.indicated_rpm',
             'stirrer.angle_rad', 'stirrer.setpoint_rpm', 'thermostat.mode',
             'thermostat.tr_setpoint_k', 'thermostat.tj_setpoint_k', 'power.enabled',
             'power.on', 'experiment.running', 'emergency.active', 'drain.angle_rad',
             'cal_heater.on'}
MONITOR_KEYS = {'time_s', 'true_reactor_c', 'reactor_c', 'jacket_c', 'stirrer_rpm',
                'conversion', 'product_yield', 'model_reaction_power_w', 'temperature_phase',
                'heater_power_w', 'cal_heater_on'}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def monitor_samples(directory, warnings):
    path = directory / 'monitor.jsonl'
    if not path.is_file():
        return []
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Monitor path is outside the run directory.')
    samples = []
    for line, text in enumerate(read_text(path).splitlines(), 1):
        try:
            row = json.loads(text)
            if not isinstance(row, dict) or not finite(row.get('time_s')):
                raise ValueError()
            # Use the FIRST sample at a clock value: later same-time records may
            # contain a control change made by a future action. Never expose it early.
            if samples and row['time_s'] == samples[-1]['time_s']:
                continue
            if samples and row['time_s'] < samples[-1]['time_s']:
                raise ValueError()
            samples.append({**{k: v for k, v in row.items() if k in MONITOR_KEYS},
                            'line': line})
        except (ValueError, TypeError):
            warnings.append(f'monitor.jsonl line {line} was skipped.')
    return samples


def update_readings(measurement, readings, controls, time_s, revision, source):
    if not isinstance(measurement, dict):
        return
    native = measurement.get('native') or {}
    if native.get('status') not in (None, 'succeeded'):
        return
    measured_at = native.get('clock', {}).get('time_s', time_s)
    for key, slot in native.get('observations', {}).items():
        if key not in READ_KEYS or not isinstance(slot, dict):
            continue
        value = slot.get('value')
        readings[key] = {'value': value, 'unit': slot.get('unit'), 'quality': slot.get('quality'),
                         'time_s': slot.get('sample_time_s', measured_at),
                         'revision': measurement.get('revision', revision),
                         'evidence_id': measurement.get('id'), 'source': source}
        if key in {'thermostat.mode', 'power.enabled', 'power.on', 'experiment.running',
                   'stirrer.setpoint_rpm', 'thermostat.tr_setpoint_k',
                   'thermostat.tj_setpoint_k', 'emergency.active', 'drain.angle_rad'}:
            controls[key] = {'value': value, 'source': 'Instrument reading', 'time_s': measured_at}


def receipt_controls(event, controls, time_s):
    """Successful receipts are authoritative for commands, not sensor measurements."""
    response = event.get('result', {}).get('native_response') or {}
    if response.get('status') != 'succeeded' or response.get('error'):
        return
    proposal = event.get('proposal') or {}
    operation = proposal.get('capability_id', '').split('::')[-1]
    args, result = proposal.get('arguments', {}), response.get('result') or {}
    updates = {}
    if operation == 'experiment.start':
        updates['experiment.running'] = True
    elif operation == 'experiment.stop':
        updates['experiment.running'] = False
    elif operation == 'stirrer.set' and 'setpoint_rpm' in result:
        updates['stirrer.setpoint_rpm'] = result['setpoint_rpm']
    elif operation in {'thermostat.set_tj', 'thermostat.set_tr'}:
        mode = operation.rsplit('_', 1)[-1]
        updates['thermostat.mode'] = mode
        if f'{mode}_setpoint_k' in result:
            updates[f'thermostat.{mode}_setpoint_k'] = result[f'{mode}_setpoint_k']
        # An accepted ramp's end target is different from its current setpoint.
        # Retain it only after a successful receipt, for the labeled visual estimate.
        if finite(args.get('end_k')):
            updates[f'thermostat.{mode}_target_k'] = args['end_k']
    # Preserve only returned values for mode/power: argument names differ between devices.
    for field, key in [('mode', 'thermostat.mode'), ('power_enabled', 'power.enabled')]:
        if field in result:
            updates[key] = result[field]
    for key, value in updates.items():
        controls[key] = {'value': value, 'source': 'Successful command receipt', 'time_s': time_s}


def build_replay(root, run_id, *, detail_index=None):
    directory = run_directory(root, run_id)
    result = read_json(directory / 'result.json')
    manifest = directory / 'manifest.json'
    task = read_json(manifest).get('task', {}) if manifest.is_file() and manifest.resolve().is_relative_to(directory) else {}
    events, warnings = read_events(directory)
    samples = monitor_samples(directory, warnings)
    times = [s['time_s'] for s in samples]
    steps, readings, controls, seen_measurements = [], {}, {}, set()
    time_s, revision = 0.0, 0
    for event in events:
        kind = event.get('kind')
        source = {'file': event['_source'], 'line': event['_line']}
        if kind == 'observation':
            observed = event.get('observations') or {}
            if finite(observed.get('simulated_seconds')):
                time_s = observed['simulated_seconds']
            revision = observed.get('revision', revision)
            measurement = observed.get('measurement') or observed.get('inspection')
            mid = measurement.get('id') if isinstance(measurement, dict) else None
            if measurement and mid not in seen_measurements:
                update_readings(measurement, readings, controls, time_s, revision, source)
                seen_measurements.add(mid)
                if steps and steps[-1]['category'] == 'action':
                    steps[-1]['readings'] = deepcopy(readings)
                    steps[-1]['controls'] = deepcopy(controls)
            continue
        sent = bool(metadata(event).get('request_sent'))
        if not sent and kind not in KINDS:
            continue
        if detail_index is not None and len(steps) == detail_index:
            return sanitize({'event': event})
        previous_time = time_s
        if kind == 'tool_result':
            response = event.get('result') or {}
            advance = response.get('advanced_seconds')
            if finite(advance) and advance >= 0:
                time_s += advance
            measurement = response.get('measurement') or {}
            native = response.get('native_response') or measurement.get('native') or {}
            clock = native.get('clock', {}).get('time_s')
            if finite(clock):
                time_s = clock
            revision = event.get('revision', revision)
            update_readings(measurement, readings, controls, time_s, revision, source)
            if measurement:
                seen_measurements.add(measurement.get('id'))
            receipt_controls(event, controls, time_s)
        category = 'model' if sent else 'action' if kind == 'tool_result' else 'result' if kind in {'agent_report', 'instrument_score'} else 'error'
        meta = metadata(event)
        model = meta.get('model_id') or meta.get('request', {}).get('model') or event.get('backend_id') or meta.get('backend_id') or ''
        monitor_index = bisect_right(times, time_s) - 1
        monitor = samples[monitor_index] if monitor_index >= 0 else None
        proposal = event.get('proposal') or {}
        steps.append({'index': len(steps), 'category': category, 'component': event.get('component'),
                      'kind': kind, 'timestamp': event.get('timestamp'), 'source': source,
                      'model': model, 'time_s': time_s, 'advance_s': max(0, time_s - previous_time),
                      'revision': revision, 'capability': proposal.get('capability_id'),
                      'arguments': proposal.get('arguments', {}),
                      'outcome': event.get('output') or event.get('decision') or event.get('report') or event.get('score'),
                      'error': event.get('error'), 'readings': deepcopy(readings), 'controls': deepcopy(controls),
                      'monitor': monitor, 'monitor_index': monitor_index,
                      'keyframe': category != 'model' or not steps})
    if detail_index is not None:
        raise FileNotFoundError('Step does not exist.')
    if steps:
        steps[-1]['keyframe'] = True
    summary = brief(directory, result)
    return sanitize({'summary': summary, 'task': task, 'steps': steps, 'samples': samples,
                     'warnings': warnings, 'supported_model': summary.get('device') == 'mt_rc1mxhfcal',
                     'model_url': '/replay/model/rc1mx.glb',
                     'truth_note': 'Simulator ground truth from the independent monitor; not a physical measurement or an agent observation.'})


def response_for(url, root, web_root, project_root):
    path = unquote(urlsplit(url).path)
    assets = {'/replay': ('replay.html', 'text/html; charset=utf-8'),
              '/replay/': ('replay.html', 'text/html; charset=utf-8'),
              '/replay/replay.css': ('replay.css', 'text/css; charset=utf-8'),
              '/replay/replay.js': ('replay.js', 'text/javascript; charset=utf-8'),
              '/replay/scene.js': ('replay-scene.js', 'text/javascript; charset=utf-8')}
    for name in VENDOR_FILES:
        assets['/replay/vendor/three/' + name] = ('vendor/three/' + name, 'text/javascript; charset=utf-8')
    try:
        if path in assets:
            name, mime = assets[path]
            file = (web_root / name).resolve()
            if not file.is_relative_to(web_root.resolve()):
                raise ValueError('Asset is outside the web directory.')
            return 200, mime, file.read_bytes()
        if path == '/replay/model/rc1mx.glb':
            file = (project_root / MODEL_DIR / 'model.glb').resolve()
            if not file.is_relative_to((project_root / MODEL_DIR).resolve()):
                raise ValueError('Model path is invalid.')
            return 200, 'model/gltf-binary', file.read_bytes()
        if path == '/api/replay/runs':
            payload = {'runs': list_runs(root)}
        elif path.startswith('/api/replay/run/'):
            match = re.fullmatch(r'/api/replay/run/([^/]+)(?:/step/(\d+))?', path)
            if not match:
                raise ValueError('Invalid replay path.')
            run_id, index = match.groups()
            payload = build_replay(root, run_id, detail_index=int(index) if index is not None else None)
        elif path.startswith(('/api/replay/', '/replay/')):
            raise FileNotFoundError()
        else:
            return None
        return 200, 'application/json; charset=utf-8', json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    except FileNotFoundError:
        code, payload = 404, {'error': 'Replay file not found. For the 3D model, run tools/fetch_replay_assets.py.'}
    except (ValueError, OSError) as exc:
        code, payload = 400, {'error': 'Replay record could not be loaded.', 'detail': type(exc).__name__}
    return code, 'application/json; charset=utf-8', json.dumps(payload).encode()
