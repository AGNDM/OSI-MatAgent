"""Harness-owned reset configuration. Never pass fixture records to a role/router."""
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from app.models import Record

ROOT = Path(__file__).resolve().parents[1]


class CaseFixture(Record):
    case_id: str
    family: Literal["process", "measurement", "acceptable", "insufficient"]
    seed: int = Field(ge=0)
    initial_liquid_kg: float = Field(ge=0)
    initial_setpoint_k: float
    target_temperature_k: float
    balance_offset_kg: float = 0
    balance_available: bool = True
    dry_mass_kg: float = 0.005
    residual_limit_kg: float = 0.0001
    position: str = "front_left"
    scheduler_seconds: float = 3600


def load_cases(split: str = "dev") -> list[CaseFixture]:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    return [CaseFixture.model_validate_json(path.read_text())
            for path in sorted((ROOT / "cases" / split).glob("*.json"))]


def case_by_id(case_id: str, split: str = "dev") -> CaseFixture:
    return next(case for case in load_cases(split) if case.case_id == case_id)
