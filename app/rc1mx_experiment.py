"""One bounded, real-agent batch experiment using the unmodified RC1mx runtime.

The host supplies initial materials and declared kinetic priors. All subsequent
instrument choices remain with the existing DeepSeek/Jev graph. Evaluator-only
composition and heat ledgers never enter agent observations or model payloads.
"""
from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path

from app.single_instrument import InstrumentTask, evaluate
from app.autonomous_instrument import AutonomousInstrumentLab, AutonomousInstrumentProfile, selection_options
from app.lab_adapter import GateRejected


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / 'cases/local/rc1mx-hydrolysis.json'
MIN_CONVERSION = 0.99


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class SceneBindingRejected(ValueError):
    code = 'scenario_boundary'


class Rc1mxExperiment(AutonomousInstrumentLab):
    # Matches the native sensor/control cycle. No native model is patched.
    await_step_seconds = 2.0

    def __init__(self, task, *, log_dir=None):
        self.monitor_ready = False
        self.hold_started_s = None
        self.samples = []
        self.boundary = None
        self.log_dir = log_dir
        if task.device != 'mt_rc1mxhfcal':
            raise SceneBindingRejected('Selected device cannot host this fixed-batch native reaction scene: ' + task.device)
        super().__init__(task)
        from asset_runtime import mt_rc1mxhfcal as runtime
        self.host.register_principal('reaction-monitor', visibility='evaluator')
        self.evaluator = self.host.session('reaction-monitor')
        setup = [
            ('reactor.charge', dict(substances=[
                dict(substance_id='water', mass_kg=0.800),
                dict(substance_id='acetic_anhydride', mass_kg=0.020)],
                temperature_k=298.15, viscosity_pa_s=0.0011,
                thermal_conductivity_w_m_k=0.55, accept_priors=True)),
            ('reaction.configure', deepcopy(runtime.REACTION_EXAMPLE)),
            ('power.set', {'enabled': True}),
        ]
        for index, (capability, arguments) in enumerate(setup):
            response = self.evaluator.invoke(instance_id=self.instance_id,
                capability_id=capability, arguments=arguments,
                request_id=f'reaction-initial-condition-{index}')
            self.host._workflow_fixture_events.append(dict(
                capability_id=capability, arguments=arguments, response=response,
                purpose='Declared initial condition before any model decision; no clock advance'))
            if response.get('status') != 'succeeded':
                raise ValueError('Initial condition rejected: ' + str(response.get('error')))
        initial = self._truth()
        self.initial_anhydride_mol = initial['truth.composition_mol'].get('acetic_anhydride', 0.0)
        self.initial_acid_mol = initial['truth.composition_mol'].get('acetic_acid', 0.0)
        if self.initial_anhydride_mol <= 0:
            raise ValueError('Missing initial reactant')
        self.monitor_ready = True
        self._initial = self._read()

    def _truth(self):
        response = self.evaluator.read(instance_id=self.instance_id, state_ids=[
            'truth.composition_mol', 'truth.reactor_k', 'truth.reaction_power_w'])
        if response.get('status') != 'succeeded':
            raise ValueError('Evaluator read failed')
        return {key: slot['value'] for key, slot in response['observations'].items()}

    def _stop(self, reason, detail):
        # First boundary is latched. Later reads cannot erase or replace it.
        if not self.termination_reason:
            self.termination_reason = reason
            self.boundary = dict(reason=reason, simulated_seconds=self.lifecycle.elapsed, detail=detail)

    def _read(self):
        self.lifecycle.calls['monitor_read'] = self.lifecycle.calls.get('monitor_read', 0) + 1
        public = super()._read()
        if not self.monitor_ready:
            return public
        truth = self._truth()
        values = {key: slot['value'] for key, slot in public['observations'].items()}
        remaining = truth['truth.composition_mol'].get('acetic_anhydride', 0.0)
        conversion = 1 - remaining / self.initial_anhydride_mol
        acid = truth['truth.composition_mol'].get('acetic_acid', 0.0)
        product_yield = (acid - self.initial_acid_mol) / (2 * self.initial_anhydride_mol)
        temp = values.get('reactor.indicated_k')
        temperature_targets = [p for p in self.task.predicates if p.state_id == 'reactor.indicated_k']
        target_band_met = bool(temperature_targets) and all(evaluate(p, public)['met'] for p in temperature_targets)
        if self.hold_started_s is None and target_band_met:
            self.hold_started_s = self.lifecycle.elapsed
        sample = dict(time_s=self.lifecycle.elapsed,
            temperature_phase='holding' if self.hold_started_s is not None else 'warmup',
            reactor_c=None if temp is None else temp-273.15,
            true_reactor_c=truth['truth.reactor_k']-273.15,
            jacket_c=None if values.get('jacket.indicated_k') is None else values['jacket.indicated_k']-273.15,
            stirrer_rpm=values.get('stirrer.indicated_rpm'),
            thermostat_mode=values.get('thermostat.mode'),
            conversion=conversion, product_yield=product_yield,
            remaining_anhydride_mol=remaining, acetic_acid_mol=acid,
            model_reaction_power_w=truth['truth.reaction_power_w'],
            emergency_active=values.get('emergency.active'),
            experiment_running=values.get('experiment.running'))
        # Keep state changes at the same clock, but omit identical read duplicates.
        if not self.samples or sample != self.samples[-1]:
            self.samples.append(sample)
            if self.log_dir:
                with (self.log_dir / 'monitor.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + '\n')
        failures = [evaluate(p, public) for p in self.task.constraints if not evaluate(p, public)['met']]
        # After first entering the target band, returning below it is a boundary,
        # rather than permission to re-enter the warm-up phase.
        if self.hold_started_s is not None:
            failures.extend(evaluate(p, public) for p in temperature_targets if not evaluate(p, public)['met'])
        if failures:
            self._stop('constraint_violated', failures)
        elif not self.pending_operation and conversion >= MIN_CONVERSION and product_yield >= MIN_CONVERSION and all(
                evaluate(p, public)['met'] for p in self.task.predicates):
            self._stop('task_completed', dict(conversion=conversion, product_yield=product_yield,
                public_predicates_met=True, evidence='Native model composition + public instrument readback'))
        elif self.lifecycle.elapsed >= self.task.max_simulated_seconds:
            self._stop('simulation_budget', {'limit_s': self.task.max_simulated_seconds})
        return public

    def validate(self, proposal, pending_operation=None):
        try:
            # Fixed batch composition is part of the experiment, not an action plan.
            if proposal.capability_id.endswith(('::dosing.start', '::reactor.discharge')):
                raise GateRejected('FIXED_BATCH_COMPOSITION')
            super().validate(proposal, pending_operation)
        except GateRejected as exc:
            self._stop('capability_boundary', {'code': exc.code, 'proposal': proposal.model_dump()})
            raise

    def execute(self, proposal):
        event = super().execute(proposal)
        if event['result'].get('error'):
            self._stop('native_boundary', event['result']['error'])
        return event

    def route_context(self, state, summary, allowed):
        summary, allowed = super().route_context(state, summary, allowed)
        # Chemical completion is independently checked; public settings alone
        # cannot authorize an early success claim.
        if self.termination_reason != 'task_completed':
            allowed = [route for route in allowed if route != 'verify']
        return summary, allowed

    def score(self, report):
        score = super().score(report)
        complete = (self.termination_reason == 'task_completed' and score['protocol_correct']
                    and score['lifecycle_resolved'] and all(p['met'] for p in score['constraints']))
        score.update(goal_met=bool(complete),
            task_success=bool(complete),
            false_success=bool(report.get('status') == 'supported' and not complete),
            experiment_monitor=self.monitor_summary(),
            completion_authority='Independent host evaluator; not an agent claim or a measured conversion')
        return score

    def monitor_summary(self):
        temperatures = [s['reactor_c'] for s in self.samples if s['reactor_c'] is not None]
        temperature_limits = {p.operator: p.value - 273.15 for p in self.task.predicates
            if p.state_id == 'reactor.indicated_k' and p.operator in {'ge', 'le'}}
        warmup_limits = {p.operator: p.value - 273.15 for p in self.task.constraints
            if p.state_id == 'reactor.indicated_k' and p.operator in {'ge', 'le'}}
        holding_temperatures = [s['reactor_c'] for s in self.samples
            if s.get('temperature_phase') == 'holding' and s['reactor_c'] is not None]
        return dict(boundary=self.boundary, stop_reason=self.termination_reason,
            interval_s=self.await_step_seconds, sample_count=len(self.samples),
            warmup_temperature_band_c=[warmup_limits.get('ge'), warmup_limits.get('le')],
            hold_started_at_s=self.hold_started_s,
            holding_temperature_range_c=([min(holding_temperatures), max(holding_temperatures)] if holding_temperatures else None),
            temperature_band_c=[temperature_limits.get('ge'), temperature_limits.get('le')], min_temperature_c=min(temperatures) if temperatures else None,
            max_temperature_c=max(temperatures) if temperatures else None,
            final=self.samples[-1] if self.samples else None,
            observation_policy='Agent chooses read timing and state_ids. Monitor samples are never agent observations.',
            completion_policy='Warm up within the global temperature bounds, then stay in the target band after first entry. Conversion and stoichiometric product yield >=99%, all public target predicates satisfied, no pending operation. No required duration, control mode, setpoint, rpm or recording state.',
            caveat='2 s instrument samples; not a continuous-time guarantee. Conversion/heat are simulator truth, not calibrated HFCal measurements.')

    def export(self, output):
        write_json(output / 'monitor-summary.json', self.monitor_summary())
        if self.samples:
            with (output / 'temperature-and-conversion.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.samples[0]))
                writer.writeheader()
                writer.writerows(self.samples)
        # Pure reads only after termination: no emergency, power-off, clock or cleanup actions.
        exports = {}
        for capability, arguments, session in [
            ('events.read', {'since_s': 0}, self.session),
            ('ledger.read', {}, self.evaluator),
            ('provenance.read', {}, self.evaluator)]:
            exports[capability] = session.invoke(instance_id=self.instance_id,
                capability_id=capability, arguments=arguments, request_id='export-'+capability)
        observation = self.session.read(instance_id=self.instance_id, state_ids=['experiment.id'])
        exp_id = observation['observations']['experiment.id']['value']
        if exp_id:
            exports['experiment.data'] = self.session.invoke(instance_id=self.instance_id,
                capability_id='experiment.data', arguments={'experiment_id': exp_id}, request_id='export-experiment')
        write_json(output / 'native-export.json', exports)


def main():
    import argparse
    from datetime import datetime
    from app.instrument_cli import run_episode
    from app.expert_config import load_plan
    from app.token_budget import BudgetedExpert, TokenLedger
    from app.openrouter_jev import OpenRouterJevBackend, OpenRouterJevOperationBackend
    from app.models import Limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='Initialize and validate only; no models or time advance')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    task = InstrumentTask.model_validate_json(TASK_PATH.read_text(encoding='utf-8'))
    if args.preflight:
        # Offline fixture validation only; real runs must select a device first.
        lab = Rc1mxExperiment(task.model_copy(update={'device': 'mt_rc1mxhfcal'}))
        assert not lab.termination_reason, lab.monitor_summary()
        assert lab.lifecycle.elapsed == 0
        assert not any('truth.' in key for key in lab.fields)
        assert not any(key.endswith(('::reaction.configure', '::reactor.charge')) for key in lab.capabilities())
        print(json.dumps(dict(preflight='passed', initial=lab.samples[-1],
            public_capabilities=len(lab.capabilities()), kinetics='upstream REACTION_EXAMPLE',
            device_choices=len(selection_options(task)['catalog']['devices']),
            no_model_calls=True, no_clock_advance=True), ensure_ascii=False, indent=2))
        return
    output = args.output or ROOT / 'results' / ('rc1mx-live-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    if output.exists():
        raise ValueError('Use a fresh output directory; no retries overwrite a run')
    plan = load_plan(ROOT / 'configs/openrouter-deepseek-v4.1-flash.json')
    plan.reference.max_tokens = 6144
    ledger = TokenLedger()
    roles = BudgetedExpert(plan.reference, ledger)
    holder = []
    def create_lab(task):
        lab = Rc1mxExperiment(task, log_dir=output)
        holder.append(lab)
        return lab
    result = run_episode(task, output, roles,
        router=OpenRouterJevBackend(ledger), operation=OpenRouterJevOperationBackend(ledger),
        profile=AutonomousInstrumentProfile(), lab_factory=create_lab, router_wait=False,
        selection_options=selection_options(task),
        limits=Limits(lab_actions=200, llm_calls=500, episode_seconds=1200, graph_steps=2400, format_repairs=0))
    write_json(output / 'task.json', task.model_dump())
    sources = ['app/rc1mx_experiment.py', 'app/autonomous_instrument.py', 'cases/local/rc1mx-hydrolysis.json',
        'app/structured_output.py', 'app/local_roles.py', 'app/expert_config.py', 'app/models.py',
        'configs/openrouter-deepseek-v4.1-flash.json',
        'app/jev_context.py', 'app/openrouter_jev.py', 'app/operation_decision.py', 'app/token_budget.py',
        'app/action_memory.py', 'app/router_context.py', 'app/local_decision.py',
        'app/graph.py', 'app/single_instrument.py', 'app/device_selection.py', 'app/instrument_cli.py',
        'vendor/chemlab/scripts/asset_runtime/mt_rc1mxhfcal.py',
        'vendor/chemlab/assets/expansion_batch04/t1_thermal/mt_rc1mxhfcal/geometry_manifest.json',
        'vendor/chemlab/assets/expansion_batch04/t1_thermal/mt_rc1mxhfcal/reaction_mass_library.json']
    write_json(output / 'source-provenance.json', {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in sources})
    if holder:
        lab = holder[0]
        lab.export(output)
        result['experiment'] = lab.monitor_summary()
        write_json(output / 'result.json', result)
    print(json.dumps(dict(output=str(output), report=result['report'], score=result['score']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
