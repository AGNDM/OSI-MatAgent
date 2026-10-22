"""Small JSON-compatible records shared by generation and evaluation."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Step:
    kind: str = "invoke"
    capability_id: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    dt_s: float = 0.0
    state_ids: list[str] | None = None
    answer: str | None = None
    actor: str = "agent"

    def to_dict(self):
        return asdict(self)


@dataclass
class Condition:
    state_id: str
    operator: str
    value: Any
    path: list[str | int] = field(default_factory=list)
    tolerance: float = 1e-6
    visibility: str = "agent"

    def to_dict(self):
        return asdict(self)


@dataclass
class DatasetConfig:
    seeds_per_family: int = 8
    seed: int = 42
    variants_per_case: int = 2
    heldout_instruments: tuple[str, ...] = ("eppendorf_cellxpertc170i",)
    heldout_fault_families: tuple[str, ...] = ("sensor_failure",)
    validation_fraction: float = 0.15
    test_fraction: float = 0.2
    template_test_fraction: float = 0.15


@dataclass
class Scenario:
    """Private scenario specification; never serialize it as a solver input."""
    family_id: str
    instrument_id: str
    domain: str
    task_type: str
    difficulty: str
    tier: str
    question: str
    options: list[dict]
    decision_kind: str = ""
    prediction_target: str = "status"
    setup_steps: list[dict] = field(default_factory=list)
    fixture_conditions: list[dict] = field(default_factory=list)
    reference_steps: list[dict] = field(default_factory=list)
    goal_conditions: list[dict] = field(default_factory=list)
    acceptable_answers: list[str] = field(default_factory=list)
    answer_conditions: dict[str, list[dict]] = field(default_factory=dict)
    causes: list[dict] = field(default_factory=list)
    scheduled_events: list[dict] = field(default_factory=list)
    probe_steps: list[dict] = field(default_factory=list)
    diagnostic_steps: list[dict] = field(default_factory=list)
    evidence_condition: str = "ordinary"
    observation_ids: list[str] | None = None
    paired_setup_steps: list[dict] | None = None
    paired_expected_answer: str | None = None
    manual: list[str] = field(default_factory=list)
    difficulty_axes: dict = field(default_factory=dict)
    risk: str = "reversible_operation"
    fault_family: str = "precondition"
    healthy_control: bool = False
    escalation_expected: bool = False
    escalation_conditions: list[dict] = field(default_factory=list)
    history_from_step: int | None = None
    budget: dict = field(default_factory=lambda: {"max_actions": 12, "max_sim_time_s": 3600.0})

