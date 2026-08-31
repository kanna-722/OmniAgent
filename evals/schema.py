from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Status = Literal["PASS", "PARTIAL", "FAIL"]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    suite: Literal["agent", "rag", "memory"]
    set_type: Literal["gold", "boundary", "failure", "adversarial"]
    input: dict[str, Any]
    expected: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    severity: Literal["normal", "high", "critical"] = "normal"
    scorer_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    run_id: str
    branch: str
    commit: str
    case_id: str
    suite: str
    set_type: str
    status: Status
    metrics: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    failure_code: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
