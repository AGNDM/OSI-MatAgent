"""Compact existing authenticated teacher-forced report fields; no scoring or inference."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

HASH = re.compile(r'[0-9a-f]{64}')


def count(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Invalid nonnegative count: ' + field)
    return value


def fraction(value, field):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('Invalid source fraction: ' + field)
    return value


def actions(source):
    """Copy the source's all-closed-attempt metrics, including unknown coverage."""
    result = {
        'observed': source['observed_original_proposals'],
        'correct': source['correct_original_proposals'],
        'accuracy': fraction(source['original_proposal_accuracy_micro'], 'original accuracy'),
        'corrections': source['reference_corrections'],
        'proposal_coverage': fraction(source['proposal_count_coverage'], 'proposal coverage'),
        'correction_coverage': fraction(source['correction_count_coverage'], 'correction coverage'),
        'known_partial_observed': count(source['known_partial_observed_original_proposals'], 'known partial proposals'),
        'known_partial_corrections': count(source['known_partial_reference_corrections'], 'known partial corrections')}
    for field in ('observed', 'correct', 'corrections'):
        if result[field] is not None:
            count(result[field], field)
    return result


def compact(report, expected_study):
    """Project only saved public report fields; validate, never calculate scores."""
    cohort = report['cohort']
    if (report.get('schema_version') != 1 or report.get('mode') != 'teacher_forced'
            or cohort.get('mode') != 'teacher_forced' or cohort.get('seed') != 0
            or cohort.get('cells') != 22 or cohort.get('cases_per_cell') != 120
            or cohort.get('declared_episode_count') != 2640
            or cohort.get('development_smokes_included') is not False
            or Path(report['study']).resolve() != Path(expected_study).resolve()):
        raise ValueError('Expected the authenticated full 22 x 120 seed-zero teacher-forced study')
    if not isinstance(report.get('generated_utc'), str) or not report.get('protocol'):
        raise ValueError('Source report timestamp/protocol is missing')
    datetime.fromisoformat(report['generated_utc'])
    provenance = report['provenance']
    for field in ('source_manifest_sha256', 'configuration_sha256', 'jobs_sha256', 'deadline_scope_sha256'):
        if not isinstance(provenance.get(field), str) or not HASH.fullmatch(provenance[field]):
            raise ValueError('Missing authenticated source hash: ' + field)
    if (provenance.get('neural_inference_performed') is not False
            or not HASH.fullmatch(provenance.get('export_helper', {}).get('sha256', ''))):
        raise ValueError('Expected saved-record reporter provenance')
    scheduler = report['scheduler']
    available = scheduler.get('status') == 'available'
    cells, names = [], set()
    mappings = {'processed': 'episode_records', 'native_completed': 'valid_native_episodes',
        'assisted_goals': 'verified_outcome_successes', 'aborted': 'aborted_episodes',
        'missing': 'missing_episode_records', 'deadline_not_started': 'deadline_not_started',
        'unusable': 'unusable_records'}
    for source in report['cells']:
        name = source['experiment']
        if (not isinstance(name, str) or not name or name in names
                or source.get('mode') != 'teacher_forced' or source.get('seed') != 0
                or source.get('expected_cases') != 120):
            raise ValueError('Cell identity, mode or declared denominator changed')
        names.add(name)
        row = {'experiment': name, 'expert': source['expert'], 'chooser': source['chooser'],
            'batch': source['batch'], 'planned': 120,
            **{key: count(source[field], field) for key, field in mappings.items()},
            'native_coverage': fraction(source['native_score']['coverage'], 'native coverage'),
            'original': actions(source['original_action_metrics_all_closed_attempts'])}
        if (row['processed'] + row['missing'] != 120
                or row['native_completed'] + row['aborted'] != row['processed']
                or row['assisted_goals'] > row['native_completed']
                or row['deadline_not_started'] > row['missing'] or row['unusable'] > row['missing']):
            raise ValueError('Cell denominator/closed-record counts do not reconcile')
        states = None
        if available:
            states = Counter()
            for case in source['cases']:
                state = case.get('slurm', {}).get('state')
                if state is not None:
                    if not isinstance(state, str) or not state:
                        raise ValueError('Invalid saved scheduler state')
                    states[state] += 1
            states = dict(states)
        row['scheduler_states'] = states
        cells.append(row)
    if len(cells) != 22:
        raise ValueError('Expected exactly 22 distinct cells')
    progress = report['progress']
    totals = {'planned': count(progress['declared_episode_count'], 'planned'),
        **{key: count(progress[field], field) for key, field in mappings.items()},
        'record_coverage': fraction(progress['attempt_record_coverage'], 'record coverage'),
        'native_coverage': fraction(progress['valid_native_episode_coverage'], 'native coverage'),
        'original': actions(report['original_action_metrics_all_closed_attempts'])}
    if totals['planned'] != 2640 or any(totals[key] != sum(row[key] for row in cells) for key in mappings):
        raise ValueError('Global and per-cell counts do not reconcile')
    scheduler_summary = {'available': available, 'source_status': scheduler.get('status'),
        'states': deepcopy(scheduler.get('states')) if available else None,
        'matched_declared_slots': scheduler.get('matched_declared_slots') if available else None}
    if available:
        for state, value in scheduler_summary['states'].items():
            count(value, 'scheduler ' + state)
    elif scheduler.get('error'):
        scheduler_summary['error'] = scheduler['error']
    return {'schema_version': 1, 'study': report['study'], 'mode': report['mode'],
        'protocol': report['protocol'], 'status': report['status'],
        'study_stopped': report['study_stopped'], 'report_generated_utc': report['generated_utc'],
        'deadline_utc': cohort['deadline_utc'],
        'outcome_semantics': report['outcome_semantics'],
        'definitions': {
            'processed': 'Usable closed episode records, including saved aborts; not scheduler COMPLETED states.',
            'assisted_goals': 'Native goals after teacher-forced reference corrections; not autonomous success.',
            'missing': 'No usable closed episode; includes queued/running/unstarted workers. Pending is not failure.',
            'original': 'Existing all-closed-attempt source metrics; auxiliary queries and saved abort decisions included. Null totals remain unknown; known partial counts are labeled.',
            'scores': 'Copied from the authenticated public report; no new accuracy, correction-rate or outcome formulas.'},
        'totals': totals, 'scheduler': scheduler_summary, 'cells': cells,
        'source': {field: deepcopy(provenance[field]) for field in
            ('source_manifest_sha256', 'configuration_sha256', 'jobs_sha256', 'deadline_scope_sha256',
             'export_helper', 'integrity_scope')}}


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('A genuine Slurm compute allocation is required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve() or args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError('Status output must be separate from its input and outside the frozen study')
    data = args.input.read_bytes()
    report = json.loads(data, parse_constant=reject_constant)
    result = compact(report, args.study)
    result.update(generated_utc=datetime.now(timezone.utc).isoformat(),
        exporter_slurm_job_id=os.environ['SLURM_JOB_ID'], exporter_compute_node=os.environ['SLURMD_NODENAME'])
    result['source'].update(public_report=str(args.input.resolve()),
        public_report_sha256=hashlib.sha256(data).hexdigest(), public_report_bytes=len(data),
        status_exporter=str(Path(__file__).resolve()),
        status_exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
    temporary = args.output.with_name('.' + args.output.name + '.' + os.environ['SLURM_JOB_ID'] + '.tmp')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(json.dumps({'output': str(args.output), 'bytes': len(encoded.encode()),
        'report_generated_utc': result['report_generated_utc'], 'processed': result['totals']['processed'],
        'assisted_goals': result['totals']['assisted_goals'], 'aborted': result['totals']['aborted'],
        'scheduler_available': result['scheduler']['available']}), flush=True)


if __name__ == '__main__':
    main()
