"""LEGACY task-specific demonstration, not the native-only agent workflow.

Paper-grounded titania substep, with an explicitly authored chemistry bridge.

Native furnace dynamics and template-derived XRD acquisition are unchanged.
No claim of calibrated reaction kinetics, physical transport, or paper replication.
"""
from copy import deepcopy
import json
import math
from uuid import uuid4

from jsonschema import Draft202012Validator
from scipy.signal import find_peaks

from app import device_registry  # makes pinned upstream runtime importable
from basic40_runtime_thermal import NaberthermL9Runtime
from basic40_runtime_measure import RigakuMiniFlex600Runtime
from basic40_templates.rigaku_miniflex600 import public_reference
from app.lab_adapter import GateRejected
from app.device_roles import DeviceProfile
from app.roles import projection

PAPER = 'https://doi.org/10.3390/ma11122364'
BRANCHES = ('thermal_history', 'scan_coverage')
CLASSIFICATION = 'synthetic template-derived acquisition; authored endpoint chemistry'
REFERENCES = {'anatase': [25.23, 37.72, 47.89, 53.77, 54.89],
              'rutile': [27.45, 36.10, 41.26, 54.35, 56.65]}


def analyze(pattern):
    """Qualitative reference-window matching, not a phase-fraction estimator."""
    points = pattern['samples']
    x = [math.degrees(p['two_theta_rad']) for p in points]
    y = [p['intensity_cps'] for p in points]
    peaks, _ = find_peaks(y, height=max(y, default=0)*0.05)
    observed = [x[int(i)] for i in peaks]
    matched = {phase: [r for r in refs if any(abs(p-r) <= .15 for p in observed)]
               for phase, refs in REFERENCES.items()}
    coverage = bool(x and x[0] <= 20.001 and x[-1] >= 59.999)
    complete = pattern['status'] == 'completed'
    return {'method': 'local maxima above 5% global maximum, +/-0.15 degree reference windows',
            'observed_peak_degrees': observed, 'matched_reference_degrees': matched,
            'acquired_range_degrees': [x[0], x[-1]] if x else [], 'point_count': len(x),
            'complete': complete, 'required_coverage': coverage,
            'qualitative_goal_met': bool(complete and coverage and len(matched['rutile']) >= 3
                                         and len(matched['anatase']) >= 2)}


class TitaniaLab:
    """Host owns endpoint fixtures; agent sees only public receipts and acquired data.

    Thermal repair uses a retained, dried sister aliquot, avoiding an unsupported
    prediction for sequential 400 C -> 600 C reprocessing of the original sample.
    """
    def __init__(self, branch, scan_timeout_s=120):
        if branch not in BRANCHES:
            raise ValueError(branch)
        self.branch = branch
        self.sample_id = 'titania-batch-001'
        self.aliquot_id = 'aliquot-001'
        self.revision = self.action_count = self.last_mutation_step = 0
        self.thermal_attempts = 0
        self.time_s = 0.
        self.scan_timeout_s = scan_timeout_s
        self.history, self.native_events, self.thermal_runs = [], [], []
        self._requests = {}
        self._inspection = self._measurement = None
        self._pattern = None
        self.scene_updates = []
        def scene_adapter(*args):
            self.scene_updates.append(deepcopy(args))
        self.furnace = NaberthermL9Runtime(ambient_temperature_K=298.15, scene_adapter=scene_adapter)
        self.xrd = RigakuMiniFlex600Runtime(scene_adapter=scene_adapter)
        for runtime, ident in [(self.furnace, 'furnace-001'), (self.xrd, 'xrd-001')]:
            runtime.create_instance(ident)
            runtime.register_principal('agent', visibility='agent', instance_ids=[ident])
        self.f = self.furnace.session('agent')
        self.x = self.xrd.session('agent')
        self.xrd.host_set_cooling('xrd-001', connection_id='fixture-chiller', ready=True)
        # Explicit reset fixture: actually execute the earlier native thermal program.
        self._heat(400 if branch == 'thermal_history' else 600, origin='reset_fixture')
        self._acquire(60 if branch == 'thermal_history' else 26, timeout_s=120)
        self.initial_analysis = analyze(self._pattern)
        self.fixture_events = deepcopy(self.native_events)
        self.native_events = []
        self.initial_thermal_count = len(self.thermal_runs)

    def _invoke(self, instrument, capability, arguments):
        session, ident = (self.f, 'furnace-001') if instrument == 'furnace' else (self.x, 'xrd-001')
        response = session.invoke(instance_id=ident, capability_id=capability,
                                  arguments=arguments, request_id=str(uuid4()))
        self.native_events.append({'time_s': self.time_s, 'instrument': instrument,
                                   'capability_id': capability, 'arguments': deepcopy(arguments),
                                   'response': deepcopy(response)})
        if response.get('status') != 'succeeded':
            raise RuntimeError(json.dumps(response))
        return response['result']

    def _advance(self, dt):
        # Furnace accepts delta; diffractometer accepts absolute time.
        self.time_s += dt
        self.furnace.step(dt)
        self.xrd.host_step(self.time_s)

    def _read(self, instrument, fields):
        session, ident = (self.f, 'furnace-001') if instrument == 'furnace' else (self.x, 'xrd-001')
        response = session.read(instance_id=ident, state_ids=fields)
        if response.get('status') != 'succeeded':
            raise RuntimeError(json.dumps(response))
        return {k: v['value'] for k, v in response['observations'].items()}

    def _heat(self, temperature_c, origin):
        # A new paper-protocol aliquot begins near room temperature.
        for _ in range(1200):
            if self._read('furnace', ['thermal.indicated_temperature_k'])['thermal.indicated_temperature_k'] <= 298.65:
                break
            self._advance(60)
        else:
            raise TimeoutError('ROOM_TEMPERATURE_START_TIMEOUT')
        start_temperature = self._read('furnace', ['thermal.indicated_temperature_k'])
        start = self.time_s
        for cap, args in [('power.set', {'enabled': True}), ('door.close', {}),
                          ('thermal.configure_program', {'segments': [{'temperature_K': temperature_c+273.15,
                              'ramp_rate_K_s': .15, 'hold_s': 7200}]}), ('thermal.start_program', {})]:
            self._invoke('furnace', cap, args)
        polls = []
        for _ in range(300):
            self._advance(60)
            state = self._read('furnace', ['program.phase', 'program.hold_elapsed_s',
                                         'thermal.indicated_temperature_k'])
            polls.append({'time_s': self.time_s, **state})
            if state['program.phase'] == 'completed':
                break
        else:
            raise TimeoutError('FURNACE_PROGRAM_TIMEOUT')
        receipt = {'aliquot_id': self.aliquot_id, 'origin': origin,
                   'temperature_c': temperature_c, 'ramp_c_min': 9, 'hold_s': 7200,
                   'start_s': start, 'start_temperature': start_temperature, 'completion': state, 'polls': polls}
        self._invoke('furnace', 'power.set', {'enabled': False})
        for _ in range(600):
            self._advance(60)
            cooled = self._read('furnace', ['thermal.indicated_temperature_k'])
            if cooled['thermal.indicated_temperature_k'] <= 313.15:
                break
        else:
            raise TimeoutError('COOLDOWN_TIMEOUT')
        receipt['transfer_temperature_k'] = cooled['thermal.indicated_temperature_k']
        receipt['end_s'] = self.time_s
        self.thermal_runs.append(receipt)
        # Authored finite endpoint lookup, never a native chemical prediction.
        self._phases = ([{'phase_id': 'anatase', 'weight_fraction': .3},
                         {'phase_id': 'rutile', 'weight_fraction': .7}] if temperature_c == 600
                        else [{'phase_id': 'anatase', 'weight_fraction': 1.0}])

    def _acquire(self, stop_deg, timeout_s):
        if not self._phases or self.thermal_runs[-1]['aliquot_id'] != self.aliquot_id:
            raise ValueError('CURRENT_ALIQUOT_NOT_PREPARED')
        self._pattern = None
        self.pattern_aliquot = None
        self.last_method = {'start_deg': 20, 'stop_deg': stop_deg, 'step_deg': .05, 'dwell_s': .05}
        for cap, args in [('power.set', {'enabled': True}),
                          ('access.set_door_angle', {'angle_rad': .8}),
                          ('slot.set_sample', {'connection_id': self.aliquot_id}),
                          ('access.set_door_angle', {'angle_rad': 0}),
                          ('xray.configure', {'voltage_v': 40000, 'current_a': .015}),
                          ('xrd.set_slit', {'slit_id': 'declared-fixture-slit'}),
                          ('xrd.configure_scan', {'start_rad': math.radians(20), 'stop_rad': math.radians(stop_deg),
                              'step_rad': math.radians(.05), 'integration_time_s': .05})]:
            self._invoke('xrd', cap, args)
        self.xrd.host_load_template_sample('xrd-001', deepcopy(self._phases))
        self.native_events.append({'time_s': self.time_s, 'instrument': 'host',
            'capability_id': 'authored.sample_handoff', 'aliquot_id': self.aliquot_id,
            'note': 'Logical transfer and template response injection; no robotics or chemical simulation.'})
        self._invoke('xrd', 'xrd.start_scan', {})
        states = [self._read('xrd', ['scan.status', 'scan.completed_points', 'scan.total_points'])]
        elapsed = 0.
        while states[-1]['scan.status'] == 'running' and elapsed < timeout_s:
            dt = min(2., timeout_s-elapsed)
            self._advance(dt)
            elapsed += dt
            states.append(self._read('xrd', ['scan.status', 'scan.completed_points', 'scan.total_points']))
        self.last_scan_states = states
        if states[-1]['scan.status'] != 'completed':
            self._invoke('xrd', 'xrd.stop_scan', {})
            raise TimeoutError('XRD_SCAN_TIMEOUT_NO_VALID_MEASUREMENT')
        self._pattern = self._invoke('xrd', 'xrd.read_template_pattern', {})
        # Upstream synthetic:false refers to reference provenance. Override to prevent mislabelling acquisition.
        self._pattern['synthetic'] = True
        self._pattern['data_classification'] = CLASSIFICATION
        self.pattern_aliquot = self.aliquot_id
        self.last_method = {'start_deg': 20, 'stop_deg': stop_deg, 'step_deg': .05, 'dwell_s': .05}

    def target(self):
        return {'sample_id': self.sample_id, 'task': 'Recover missing rutile evidence in a titania batch.',
                'goal': 'Obtain fresh complete 20-60 degree XRD with >=3 rutile and >=2 anatase reference peaks.',
                'protocol': {'temperature_c': 600, 'ramp_c_min': 9, 'hold_s': 7200},
                'alarm': 'Expected rutile reflections were not confirmed in the earlier scan.',
                'scope': CLASSIFICATION,
                'policy': 'Inspect delivered thermal history and scan method. Preserve a correctly processed aliquot. '
                          'If the wrong thermal program was delivered, prepare a retained dried sister aliquot using the reference protocol.'}

    def capabilities(self):
        descriptions = {
            'workflow.inspect': 'Read delivered thermal history, previous scan method and public acquired features.',
            'thermal.prepare_reference': 'Consume a retained dried sister aliquot; execute native 600 C, 9 C/min, 2 h program and cool below 313.15 K. Invalidates previous scan. Logical loading/transfer and endpoint chemistry are authored.',
            'xrd.acquire': 'Mount current aliquot logically, configure native Cu Ka XRD, start, await completion and read acquired pattern. No valid measurement on timeout.'}
        return {name: {'arguments': {'stop_deg': {}} if name == 'xrd.acquire' else {},
                       'input_schema': {'type': 'object', 'properties': {'stop_deg': {'type': 'number', 'enum': [26, 60]}} if name == 'xrd.acquire' else {},
                                        'required': ['stop_deg'] if name == 'xrd.acquire' else [], 'additionalProperties': False},
                       'source': descriptions[name]} for name in descriptions}

    def _snapshot(self):
        thermal = [{k: v for k, v in r.items() if k != 'polls'} for r in self.thermal_runs]
        current = next((r for r in reversed(thermal) if r['aliquot_id'] == self.aliquot_id), None)
        current_pattern = bool(self._pattern and self.pattern_aliquot == self.aliquot_id)
        return {'id': f'obs-{self.action_count}', 'sample_id': self.sample_id,
                'aliquot_id': self.aliquot_id, 'current_thermal_run': current, 'revision': self.revision,
                'captured_at_step': self.action_count, 'quality': 'valid',
                'thermal_history': thermal, 'scan_method': self.last_method if current_pattern else None,
                'pattern_aliquot_id': self.pattern_aliquot if current_pattern else None,
                'analysis': analyze(self._pattern) if current_pattern else None,
                'data_classification': CLASSIFICATION}

    def observe(self):
        return deepcopy({'revision': self.revision, 'sample_id': self.sample_id,
                         'inspection': self._inspection, 'measurement': self._measurement,
                         'last_mutation_step': self.last_mutation_step})

    def validate(self, p, pending_operation=None):
        digest = json.dumps(p.model_dump(), sort_keys=True)
        if p.sample_id != self.sample_id:
            raise GateRejected('WRONG_SAMPLE')
        if p.request_id in self._requests:
            if self._requests[p.request_id][0] != digest:
                raise GateRejected('REQUEST_CONFLICT')
            return
        if p.expected_revision != self.revision:
            raise GateRejected('STALE_PROPOSAL')
        if pending_operation:
            raise GateRejected('PENDING_OPERATION')
        cap = self.capabilities().get(p.capability_id)
        if cap is None or list(Draft202012Validator(cap['input_schema']).iter_errors(p.arguments)):
            raise GateRejected('INVALID_CAPABILITY_OR_ARGUMENT')
        if p.capability_id != 'workflow.inspect' and not self._inspection:
            raise GateRejected('INSPECTION_REQUIRED')
        if p.capability_id == 'xrd.acquire' and (not self._phases or
                self.thermal_runs[-1]['aliquot_id'] != self.aliquot_id):
            raise GateRejected('CURRENT_ALIQUOT_NOT_PREPARED')

    @staticmethod
    def action_cost(p):
        return 2 if p.capability_id == 'xrd.acquire' else 1

    def execute(self, p):
        self.validate(p)
        if p.request_id in self._requests:
            return deepcopy(self._requests[p.request_id][1])
        self.action_count += 1
        cap = p.capability_id
        start = len(self.native_events)
        thermal_count = len(self.thermal_runs)
        result = {}
        if cap == 'workflow.inspect':
            self._inspection = self._snapshot()
            result['inspection'] = self._inspection
        else:
            self.revision += 1
            self.last_mutation_step = self.action_count
            self._measurement = None
            try:
                if cap == 'thermal.prepare_reference':
                    self.thermal_attempts += 1
                    self._phases = None
                    self.aliquot_id = 'aliquot-'+str(len(self.thermal_runs)+1).zfill(3)
                    self._pattern = None
                    self._heat(600, origin='workflow_retained_dried_sister_aliquot')
                else:
                    self._acquire(p.arguments['stop_deg'], self.scan_timeout_s)
                    self.action_count += 1  # separately counted acquisition readback
                    self._measurement = self._snapshot()
                    result['measurement'] = self._measurement
            except TimeoutError as exc:
                result['error'] = {'code': str(exc)}
            # Refresh public processing state after every completed/failed macro.
            self._inspection = self._snapshot()
        result['native_events'] = deepcopy(self.native_events[start:])
        if cap == 'thermal.prepare_reference' and len(self.thermal_runs) > thermal_count:
            result['thermal_run'] = deepcopy(self.thermal_runs[-1])
        if cap == 'xrd.acquire':
            result['scan_states'] = deepcopy(self.last_scan_states)
        event = {'step': self.action_count, 'proposal': p.model_dump(), 'revision': self.revision,
                 'source': CLASSIFICATION, 'result': deepcopy(result)}
        self.history.append(event)
        self._requests[p.request_id] = (json.dumps(p.model_dump(), sort_keys=True), deepcopy(event))
        return event

    def score(self, report):
        m = self._measurement
        # Re-read native acquired points, independent of the expert's answer and cached analysis.
        raw = self._invoke('xrd', 'xrd.read_template_pattern', {}) if m else None
        acquired = bool(raw and analyze(raw)['qualitative_goal_met'])
        fresh = bool(m and m['quality'] == 'valid' and m['sample_id'] == self.sample_id and m['aliquot_id'] == self.pattern_aliquot
                     and m['revision'] == self.revision and m['captured_at_step'] > self.last_mutation_step)
        cited = bool(m and m['id'] in report.get('evidence_ids', []))
        correct_policy = (self.thermal_attempts == 0 if self.branch == 'scan_coverage'
                          else len(self.thermal_runs) > self.initial_thermal_count)
        supported = report.get('status') == 'supported'
        ok = acquired and fresh and cited and correct_policy
        return {'task_success': bool(ok and supported), 'acquired_phase_evidence': acquired,
                'fresh_identity_matched': fresh, 'evidence_cited': cited,
                'branch_policy_met': correct_policy, 'false_success': bool(supported and not ok),
                'native_call_count': len(self.native_events), 'simulated_seconds': self.time_s,
                'scope': CLASSIFICATION}


class TitaniaProfile(DeviceProfile):
    candidate_prompt = (
        'Propose next-operation candidates for the CURRENT titania episode. Read operation_context and '
        'observations.inspection first. If inspection_present is false at the initial state, propose only '
        'workflow.inspect. Otherwise the sample HAS ALREADY BEEN INSPECTED: use the delivered thermal history '
        'and scan method now supplied; do not repeat inspection of unchanged valid evidence. '
        'current_thermal_run identifies the CURRENT aliquot and its completed delivered program. Historical '
        'thermal_history entries for other aliquots must not override that current receipt. Compare it with '
        'target.protocol; a short XRD range cannot establish absent rutile. '
        'Propose schema-valid next operations justified by current evidence and target preservation policy. '
        'Explain alternative benefits and constraints briefly, citing actual observation IDs. '
        'After reference preparation, propose fresh XRD acquisition rather than more heat or inspection. '
        'Do not declare an absent inspection when operation_context.inspection_present is true. '
        'Keep each rationale and intent string to one brief sentence; do not cite retrieval document IDs.'
    )

    @staticmethod
    def validate_role_contract(role, record, payload):
        DeviceProfile.validate_role_contract(role, record, payload)
        if role == 'diagnostic' and record.intent and record.intent.capability_id == 'workflow.inspect':
            obs = payload['observations']
            inspection = obs.get('inspection')
            if (inspection and inspection['quality'] == 'valid' and inspection['revision'] == obs['revision']
                    and not payload.get('last_error')):
                raise ValueError('Current inspection is already valid and current; propose a next operation using it, not redundant inspection.')

    projection = staticmethod(projection)
    prompts = {
        'diagnostic': 'Diagnose missing rutile evidence using public thermal_history and scan_method. First workflow.inspect. '
                      'If the current aliquot was processed at 400 C instead of the reference 600 C, thermal.prepare_reference uses a fresh retained dried sister aliquot. '
                      'If the current aliquot already completed the 600 C reference protocol, preserve it. '
                      'After correct processing obtain xrd.acquire with stop_deg 60, covering 20-60 degrees. '
                      'A 20-26 scan cannot exclude rutile. Never repeat heat on a correctly processed aliquot. '
                      'Cite only observations.inspection.id or measurement.id; do not cite document IDs. Return schema-valid hypothesis and intent.',
        'procedure': DeviceProfile.prompts['procedure'],
        'verification': 'Verify qualitative biphasic XRD only. Require valid fresh observations.measurement, matching sample/aliquot, '
                        'revision equal observations.revision, captured_at_step > last_mutation_step, complete native scan, '
                        'required_coverage true and >=3 matched rutile plus >=2 matched anatase reference peaks. '
                        'If all hold return supported citing measurement.id. Missing evidence means unresolved; complete adequate evidence failing criteria means contradicted. '
                        'Do not infer quantitative fractions or physical synthesis success.',
        'deliberation': DeviceProfile.prompts['deliberation']}

    @staticmethod
    def retrieve(role):
        return [{'id': 'paper-protocol', 'source': PAPER+'; Section 2.2, Table 2',
                 'text': 'Dried TiO2-1 was split into sister aliquots. 9 C/min ramps, 2 h holds: 400 C yielded anatase; 600 C yielded anatase and rutile. Valid only for this paper feedstock. The fixture uses an authored endpoint lookup, not kinetics.'},
                {'id': 'public-xrd-reference', 'source': 'Pinned RRUFF-derived MiniFlex library',
                 'text': 'Qualitative reference peaks in degrees: '+json.dumps(REFERENCES)+'. A scan ending at 26 degrees omits the listed rutile peaks. Pattern is template-derived synthetic data. No quantitative fraction inference.'}]


class ScriptedTitaniaRoles:
    is_llm = False
    backend_id = 'scripted-titania-integration-test-double-v1'

    def invoke(self, role, prompt, payload, schema):
        if role == 'diagnostic':
            inspection = payload['observations']['inspection']
            if not inspection:
                cap, args = 'workflow.inspect', {}
            elif inspection['thermal_history'][-1]['temperature_c'] != 600:
                cap, args = 'thermal.prepare_reference', {}
            else:
                cap, args = 'xrd.acquire', {'stop_deg': 60}
            value = {'hypotheses': [{'cause': 'test_policy_from_public_evidence',
                       'supporting_ids': [inspection['id']] if inspection else [], 'contradicting_ids': []}],
                     'intent': {'capability_id': cap, 'arguments': args,
                                'discriminating_observation': 'Read delivered history and acquired peaks.',
                                'postcondition': 'Acquire adequate qualitative phase evidence.'}}
        elif role == 'procedure':
            value = {k: v for k, v in payload['intent'].items() if k != 'discriminating_observation'}
            value.update({k: payload[k] for k in ['sample_id', 'expected_revision', 'request_id']})
        elif role == 'verification':
            m = payload['observations']['measurement']
            value = {'status': 'supported' if m['analysis']['qualitative_goal_met'] else 'contradicted',
                     'evidence_ids': [m['id']], 'missing_evidence': []}
        else:
            value = {'backend_id': self.backend_id, 'choice': 'prepare_action' if payload['summary']['has_intent'] else
                     'verify' if payload['summary']['has_measurement'] else 'diagnose'}
        from app.models import OperationOptions
        if role == 'diagnostic' and schema is OperationOptions:
            from app.operation_decision import scripted_options
            return schema.model_validate(scripted_options(value, payload)).model_dump()
        return schema.model_validate(value).model_dump()
