from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

Route = Literal["diagnose", "prepare_action", "verify", "deliberate"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Intent(Record):
    capability_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    discriminating_observation: str
    postcondition: str


class Hypothesis(Record):
    cause: str
    supporting_ids: list[str]
    contradicting_ids: list[str]


class HypothesisSet(Record):
    hypotheses: list[Hypothesis]
    intent: Intent | None


class ActionProposal(Record):
    sample_id: str
    expected_revision: int = Field(ge=0)
    capability_id: str
    arguments: dict[str, Any]
    request_id: str = Field(min_length=1)
    postcondition: str


class VerificationReport(Record):
    status: Literal["supported", "unresolved", "contradicted"]
    evidence_ids: list[str]
    missing_evidence: list[str]


class RouteDecision(Record):
    backend_id: str
    choice: Route
    probabilities: dict[str, float] | None = None
    latency_ms: float = Field(default=0, ge=0)
    fallback_used: bool = False


class Limits(Record):
    lab_actions: int = Field(default=6, ge=0)
    llm_calls: int = Field(default=12, ge=0)
    episode_seconds: float = Field(default=180, gt=0)
    graph_steps: int = Field(default=60, ge=1)
    format_repairs: int = Field(default=1, ge=0, le=1)


class LabState(TypedDict, total=False):
    episode_id: str
    sample_id: str
    observation_revision: int
    observations: dict
    capabilities: dict
    target: dict
    hypotheses: list[dict]
    evidence_refs: list[str]
    intent: dict | None
    proposal: dict | None
    pending_operation: dict | None
    verification: dict | None
    route: str
    action_count: int
    llm_call_count: int
    role_call_count: int
    deadline: float
    terminal_reason: str | None
    gate_result: str
    stop_result: str
    last_error: dict | None
    report: dict
