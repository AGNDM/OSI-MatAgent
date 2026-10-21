"""Menu membership guards around the unchanged sequential native evaluator."""
from __future__ import annotations

import copy
from functools import lru_cache

from instrument_bench.adapters import INSTRUMENTS, make_adapter
from stepwise_bench.engine import World, _contract
from stepwise_bench.policies import ResumeMismatchError, zero_usage
from stepwise_expanded.engine import ReferencePolicy as NativeReferencePolicy
from stepwise_expanded.engine import run_case as native_run_case
from stepwise_expanded.tasks import build_tasks

from .menus import MenuBuilder, build_preset_bank


class MenuProtocolError(ResumeMismatchError):
    """A menu/evaluator mismatch is infrastructure failure, not a model error."""


def public_contracts(seed=42):
    return {machine: _contract(make_adapter(machine, seed)) for machine in INSTRUMENTS}


@lru_cache(maxsize=4)
def global_menus(seed=42):
    return MenuBuilder(build_preset_bank(build_tasks(seed), public_contracts(seed)))


def canonical_action(action, selected=None):
    if not isinstance(action, dict):
        return None
    kind = action.get("kind")
    if kind in {"finish", "escalate"}:
        return {"kind": kind}
    if kind == "wait":
        return {"kind": kind, "dt_s": action.get("dt_s")}
    result = {"kind": kind, "instrument_id": action.get("instrument_id", selected)}
    if kind == "invoke":
        result.update(capability_id=action.get("capability_id"), arguments=copy.deepcopy(action.get("arguments", {})))
    elif kind == "read":
        slots = action.get("state_ids")
        result["state_ids"] = sorted(slots) if isinstance(slots, list) else slots
    return result


def offered_action_path(menus, view, history, action):
    target = canonical_action(action, view.get("selected_instrument"))
    if target is None:
        return None
    for index, option in enumerate(menus.action_options(view, history)):
        if "action" in option:
            if canonical_action(option["action"], view.get("selected_instrument")) == target:
                return {"command_index": index, "command_text": option["text"], "argument_index": None}
            continue
        capability = option.get("capability_id")
        relevant = ((target["kind"] == "invoke" and capability == target.get("capability_id"))
                    or (target["kind"] == "read" and capability == "__read__")
                    or (target["kind"] == "wait" and capability == "__wait__"))
        if not relevant:
            continue
        for parameter_index, preset in enumerate(menus.argument_options(view, history, option)):
            if canonical_action(preset["action"], view.get("selected_instrument")) == target:
                return {"command_index": index, "command_text": option["text"],
                        "argument_index": parameter_index, "argument_text": preset["text"]}
    return None


def action_is_offered(menus, public_view, history, action):
    return offered_action_path(menus, public_view, history, action) is not None


class MenuReferencePolicy:
    """Privileged author diagnostic constrained to the same public choices."""
    privileged = True

    def __init__(self, case, menu_builder):
        self.reference = NativeReferencePolicy(case)
        self.menu_builder = menu_builder
        self.last_action_usage = zero_usage()
        self.identity = {"method": "menu-reference", "privileged": True}

    def next_action(self, view, history):
        action = self.reference.next_action(view, history)
        path = offered_action_path(self.menu_builder, view, history, action)
        if path is None:
            raise MenuProtocolError("Reference action is absent from the shared public menu")
        self.last_action_usage = {**zero_usage(), "menu_reference_path": path}
        return action


class WrongMenuPolicy:
    """Privileged negative diagnostic: choose a real, offered wrong machine."""
    privileged = True

    def __init__(self, case, menu_builder):
        self.target = case["instrument_id"]
        self.menu_builder = menu_builder
        self.last_action_usage = zero_usage()
        self.identity = {"method": "offered-wrong-machine", "privileged": True,
                         "purpose": "Check semantic mistakes without formatting errors"}

    def next_action(self, view, history):
        action = next(option["action"] for option in self.menu_builder.action_options(view, history)
                      if option.get("action", {}).get("kind") == "select_instrument"
                      and option["action"]["instrument_id"] != self.target)
        self.last_action_usage = {**zero_usage(), "menu_action": copy.deepcopy(action)}
        return copy.deepcopy(action)


def run_case(case, policy, mode="teacher_forced", on_step=None, *, menu_builder=None):
    menus = menu_builder or getattr(policy, "menu_builder", None) or global_menus(case.get("seed", 42))
    contexts = []
    menu_proofs = []

    class GuardedPolicy:
        privileged = bool(getattr(policy, "privileged", False))
        last_action_usage = {}

        def next_action(self, view, history):
            contexts.append(copy.deepcopy((view, history)))
            self.last_action_usage = zero_usage()
            try:
                action = policy.next_action(view, history)
            finally:
                self.last_action_usage = copy.deepcopy(getattr(policy, "last_action_usage", {}))
            if not action_is_offered(menus, view, history, action):
                raise MenuProtocolError("Policy returned an action not offered by its public menu")
            return action

    guarded = GuardedPolicy()

    def check_step(record):
        view, history = contexts[-1]
        executed = record.get("executed_action")
        path = offered_action_path(menus, view, history, executed) if executed is not None else None
        if executed is not None and path is None:
            raise MenuProtocolError("Evaluator correction was not an offered public choice")
        proof = {"executed_menu_path": path,
                 "proposed_menu_path": offered_action_path(menus, view, history, record.get("proposed_action")),
                 "top_level_choices": len(menus.action_options(view, history))}
        menu_proofs.append(proof)
        if on_step:
            on_step({**copy.deepcopy(record), **copy.deepcopy(proof)})

    report = native_run_case(case, guarded, mode=mode, on_step=check_step)
    for record, proof in zip(report["decision_records"], menu_proofs):
        record.update(proof)
    report["interface"] = "public machine / command / parameter choices"
    report["menu_membership_verified"] = True
    report["menu_decisions"] = sum(len(record.get("usage", {}).get("selection_events", []))
                                    for record in report["decision_records"])
    report["format_generation_required"] = False
    report["limitations"].append("Commands use finite shared parameter presets; numeric and string spaces are not exhaustive.")
    return report
