from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ActionCandidate(Record):
    candidate_id: str = Field(min_length=1, max_length=64)
    intent: Intent
    rationale: str = Field(min_length=1)
    evidence_ids: list[str]


class OperationOptions(Record):
    """Diagnostic alternatives, not an already selected action."""
    hypotheses: list[Hypothesis]
    candidates: list[ActionCandidate] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def distinct_candidates(self):
        import json
        ids = [c.candidate_id for c in self.candidates]
        actions = [json.dumps([c.intent.capability_id, c.intent.arguments], sort_keys=True, allow_nan=False)
                   for c in self.candidates]
        if len(set(ids)) != len(ids) or len(set(actions)) != len(actions):
            raise ValueError("Candidate IDs and capability/argument pairs must be distinct")
        return self


class OperationDecision(Record):
    backend_id: str
    choice: str = Field(min_length=1)
    probabilities: dict[str, float]
    latency_ms: float = Field(default=0, ge=0)


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
    candidates: list[dict]
    operation_decision: dict | None
    operation_revision: int | None
    operation_call_count: int
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
