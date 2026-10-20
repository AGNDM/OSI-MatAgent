"""Pinned, stdlib-only literature-operation feasibility audit (no LLM calls).

Raw files stay ignored under results/. Native smoke execution is optional and
checks empty-furnace protocol completion, never synthesis/product success.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import lzma
import math
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOLID_COMMIT = 'fca0a994ddb36b37d39b803e8a21880aaecb1d71'
SOLUTION_COMMIT = '66cd72653eadab253ec04ca86a4901eb5329c022'
SOURCES = {
    'solid_2020': ('text-mined-synthesis_public--solid-state_dataset_20200713.json.xz',
        f'https://raw.githubusercontent.com/CederGroupHub/text-mined-synthesis_public/{SOLID_COMMIT}/solid-state_dataset_20200713.json.xz'),
    'solgel_2020': ('text-mined-synthesis_public--sol-gel_dataset_20200713.json.xz',
        f'https://raw.githubusercontent.com/CederGroupHub/text-mined-synthesis_public/{SOLID_COMMIT}/sol-gel_dataset_20200713.json.xz'),
    'solution': ('text-mined-solution-synthesis_public--solution-synthesis_dataset_2021-8-5.json.zip',
        f'https://raw.githubusercontent.com/CederGroupHub/text-mined-solution-synthesis_public/{SOLUTION_COMMIT}/solution-synthesis_dataset_2021-8-5.json.zip'),
}
META = {9722159: 3, 16583387: 4}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    compressed = path.read_bytes()
    if path.suffix == '.xz':
        payload = lzma.decompress(compressed)
    elif path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX/')]
            if len(names) != 1:
                raise ValueError('Expected exactly one JSON member')
            payload = z.read(names[0])
    else:
        payload = compressed
    data = json.loads(payload)
    return (data['reactions'] if isinstance(data, dict) else data), payload, compressed


def scalar(field, kind):
    """Accept only one explicit scalar, with equal non-null min/max and known units.

    Multiple values, ranges, bounds and missing units are deliberately rejected.
    This is an extraction-level screen, not validation against the source paper.
    """
    if isinstance(field, list):
        if len(field) != 1:
            return None
        field = field[0]
    if not isinstance(field, dict):
        return None
    values = field.get('values')
    if not isinstance(values, list) or len(values) != 1:
        return None
    v = values[0]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    if field.get('min_value') != v or field.get('max_value') != v:
        return None
    unit = field.get('units')
    if kind == 'temperature':
        if unit in ('°C', 'C', '℃'):
            return v + 273.15
        if unit == 'K':
            return float(v)
    if kind == 'time':
        multiplier = {'s': 1, 'sec': 1, 'min': 60, 'm': 60, 'h': 3600, 'hr': 3600}.get(unit)
        if multiplier:
            return v * multiplier
    return None


def parameters(operation):
    c = operation.get('conditions') or {}
    return (scalar(c.get('temperature', c.get('heating_temperature')), 'temperature'),
            scalar(c.get('time', c.get('heating_time')), 'time'),
            c.get('atmosphere', c.get('heating_atmosphere')))


def assess(operation):
    t, h, atmosphere = parameters(operation)
    point = t is not None and h is not None
    numeric = bool(point and 273.15 <= t <= 1373.15 and 0 <= h <= 86400)
    # Air token establishes gas identity only, not flow, pressure or apparatus.
    explicit_air = atmosphere == ['air'] or atmosphere == 'air'
    return {'temperature_K': t, 'extracted_time_s': h, 'atmosphere': atmosphere,
            'explicit_scalar_pair': point, 'numeric_envelope': numeric,
            'explicit_air_numeric_screen': numeric and explicit_air}


def native_smoke(item):
    sys.path.insert(0, str(ROOT))
    from app import device_registry  # noqa: F401: registers pinned upstream path
    from basic40_runtime_thermal import NaberthermL9Runtime
    runtime = NaberthermL9Runtime(scene_adapter=lambda *args: None)
    runtime.create_instance('audit-furnace')
    runtime.register_principal('audit-agent', visibility='agent', instance_ids=['audit-furnace'])
    session = runtime.session('audit-agent')
    events = []
    for cap, args in [('power.set', {'enabled': True}), ('door.close', {}),
                      ('thermal.configure_program', {'segments': [{
                          'temperature_K': item['temperature_K'], 'hold_s': item['extracted_time_s'],
                          'ramp_rate_K_s': 0.15}]}), ('thermal.start_program', {})]:
        response = session.invoke(instance_id='audit-furnace', capability_id=cap,
                                  arguments=args, request_id=f'audit-{len(events)}')
        events.append({'capability': cap, 'arguments': args, 'response': response})
        if response['status'] != 'succeeded':
            raise RuntimeError(response)
    fields = ['program.phase', 'program.hold_elapsed_s', 'thermal.indicated_temperature_k']
    state = None
    for tick in range(1, 2001):
        runtime.step(60)
        read = session.read(instance_id='audit-furnace', state_ids=fields)
        if read['status'] != 'succeeded':
            raise RuntimeError(read)
        state = {k: v['value'] for k, v in read['observations'].items()}
        if state['program.phase'] == 'completed':
            break
    completed = (state['program.phase'] == 'completed'
                 and state['program.hold_elapsed_s'] >= item['extracted_time_s'])
    return {'source': item, 'authored_assumptions': {'ambient_K': 298.15,
            'ramp_rate_K_s': .15,
            'extracted_time_interpretation': 'generic operation time used as post-target hold; not source-verified',
            'scene_callback': 'no-op callback; no geometry or rendered scene validation',
            'gas_flow': 'not modeled; source paragraph mentions flowing air', 'initial_state': 'new powered-off empty furnace, closed door'},
            'native_commands': events, 'final_public_state': state,
            'simulation_elapsed_s': tick * 60, 'protocol_completed': completed,
            'chemical_outcome_label': None,
            'scope': 'isolated extracted thermal parameters; empty-furnace dynamics; not reaction reproduction'}


def audit(raw, smoke):
    out = {'audit_version': 1, 'selection_rule': '12 lowest SHA256(seed=osi-dataset-audit-v1 + row index), within each licensed corpus',
           'screen_definition': 'HeatingOperation only; exact scalar T/time; 0..1100 C; 0..24h; explicit air token only (flow/pressure unconstrained); no claim of full recipe compatibility',
           'corpora': {}, 'native_smoke': []}
    smoke_items = []
    for name, (filename, url) in SOURCES.items():
        rows, payload, compressed = load(raw / filename)
        counts = Counter()
        operations = Counter()
        dois = Counter(r.get('doi') for r in rows if r.get('doi'))
        duplicate_sources = Counter((r.get('doi'), r.get('paragraph_string')) for r in rows)
        selected = set(sorted(range(len(rows)), key=lambda i: digest(f'osi-dataset-audit-v1:{i}'.encode()))[:12])
        sample = []
        for index, row in enumerate(rows):
            row_flags = Counter()
            derived = []
            for oi, op in enumerate(row.get('operations', [])):
                operations[op['type']] += 1
                if op['type'] != 'HeatingOperation':
                    continue
                counts['heating_operations'] += 1
                a = assess(op)
                for flag in ['explicit_scalar_pair', 'numeric_envelope', 'explicit_air_numeric_screen']:
                    counts['operations_' + flag] += int(a[flag])
                    row_flags[flag] |= int(a[flag])
                item = {'row_index': index, 'operation_index': oi, 'doi': row.get('doi'),
                        'row_sha256': digest(json.dumps(row, sort_keys=True, ensure_ascii=True).encode()),
                        'target_formula': row.get('target', {}).get('material_formula'), **a}
                derived.append(item)
                if (name == 'solution' and a['explicit_air_numeric_screen']
                    and 298.15 < a['temperature_K'] <= 873.15 and 0 < a['extracted_time_s'] <= 36000
                    and not any(s['corpus'] == name for s in smoke_items)):
                    smoke_items.append({'corpus': name, **item})
            for flag, val in row_flags.items():
                counts['records_' + flag] += val
            if index in selected and name == 'solution':
                sample.append({'row_index': index, 'doi': row.get('doi'),
                    'row_sha256': digest(json.dumps(row, sort_keys=True, ensure_ascii=True).encode()),
                    'operation_types_in_extracted_order': [o['type'] for o in row.get('operations', [])],
                    'thermal_operations': derived})
        counts['records_with_quantities'] = sum(bool(r.get('quantities')) for r in rows)
        counts['truncated_paragraph_records'] = sum('<...>' in r.get('paragraph_string', '') for r in rows)
        metadata = None
        if name == 'solution':
            article = 9722159 if name == 'solid_original' else 16583387
            meta = json.loads((raw / f'figshare-{article}-v{META[article]}.json').read_text())
            metadata = {'article_id': article, 'version': META[article], 'license': meta['license'],
                        'expected_payload_md5': meta['files'][0]['computed_md5']}
            if hashlib.md5(payload).hexdigest() != metadata['expected_payload_md5']:
                raise ValueError(f'{name}: Figshare checksum mismatch')
        out['corpora'][name] = {'filename': filename, 'url': url, 'bytes': len(compressed),
             'sha256': digest(compressed), 'payload_sha256': digest(payload),
             'payload_md5': hashlib.md5(payload).hexdigest(), 'records': len(rows),
             'unique_dois': len(dois), 'max_records_per_doi': max(dois.values()),
             'doi_excerpt_groups': len(duplicate_sources),
             'records_in_repeated_doi_excerpt_groups': sum(v for v in duplicate_sources.values() if v > 1),
             'operation_counts': dict(sorted(operations.items())), 'counts': dict(sorted(counts.items())),
             'figshare_license_and_checksum': metadata,
             'distribution_status': 'CC BY 4.0 and payload MD5 verified' if metadata else 'updated release license not established; aggregate-only audit',
             'deterministic_12_record_sample': sorted(sample, key=lambda s: s['row_index'])}
    if smoke:
        from app.device_registry import get_device, open_native, public_capabilities
        out['native_contracts'] = {}
        for device in ['binder_fd056', 'ika_plate_rctdigital', 'eppendorf_5910ri', 'nabertherm_l9']:
            _, _, _, description = open_native(get_device(device))
            out['native_contracts'][device] = {name: cap['input_schema'] for name, cap in public_capabilities(description).items()}
        out['native_smoke'] = [native_smoke(i) for i in smoke_items]
        if not all(r['protocol_completed'] for r in out['native_smoke']):
            raise RuntimeError('Native smoke did not complete')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'results/dataset-feasibility-v1/raw')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/evidence/dataset-feasibility-audit.json')
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--native-smoke', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.fetch:
        args.raw.mkdir(parents=True, exist_ok=True)
        jobs = list(SOURCES.values()) + [(f'figshare-{i}-v{v}.json', f'https://api.figshare.com/v2/articles/{i}/versions/{v}') for i, v in META.items()]
        for filename, url in jobs:
            path = args.raw / filename
            if not path.exists():
                with urllib.request.urlopen(url, timeout=180) as response:
                    path.write_bytes(response.read())
    result = audit(args.raw, args.native_smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')
    for name, corpus in result['corpora'].items():
        print(name, corpus['records'], json.dumps(corpus['counts']))
    print('native smoke:', [r['protocol_completed'] for r in result['native_smoke']])


if __name__ == '__main__':
    main()
