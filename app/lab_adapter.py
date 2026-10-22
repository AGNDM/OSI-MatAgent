"""Structural boundary between the shared graph and evaluator-owned instruments."""
from typing import Any, Protocol

from app.models import ActionProposal


class GateRejected(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class LabAdapter(Protocol):
    sample_id: str
    revision: int
    action_count: int

    def capabilities(self) -> dict[str, Any]: ...
    def target(self) -> dict[str, Any]: ...
    def observe(self) -> dict[str, Any]: ...
    def validate(self, proposal: ActionProposal, pending_operation=None): ...
    def execute(self, proposal: ActionProposal) -> dict[str, Any]: ...
