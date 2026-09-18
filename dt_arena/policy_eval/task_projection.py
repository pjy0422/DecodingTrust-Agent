from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .episode import TaskSnapshot


@dataclass(frozen=True)
class ProjectionPolicy:
    expose_task_instruction: bool = True
    expose_threat_model: bool = True
    expose_additional_information: bool = True


@dataclass(frozen=True)
class PolicyTaskSpec:
    """Sanitized task state that may cross the RL trust boundary."""

    task_id: str | None
    domain: str | None
    malicious_goal: str
    task_instruction: str | list[str] | None = None
    threat_model: str | None = None
    additional_information: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_task(snapshot: TaskSnapshot, *, policy: ProjectionPolicy | None = None) -> PolicyTaskSpec:
    if policy is None:
        policy = ProjectionPolicy()

    attack = snapshot.attack_config
    if attack is None:
        raise ValueError("DTAP RL requires an Attack section")

    malicious_goal = getattr(attack, "malicious_goal", None)
    if not malicious_goal or not str(malicious_goal).strip():
        raise ValueError("DTAP RL requires Attack.malicious_goal")

    task_cfg = snapshot.task_config
    return PolicyTaskSpec(
        task_id=getattr(task_cfg, "task_id", None),
        domain=getattr(task_cfg, "domain", None),
        malicious_goal=str(malicious_goal).strip(),
        task_instruction=(getattr(task_cfg, "original_instruction", None) if policy.expose_task_instruction else None),
        threat_model=(getattr(attack, "threat_model", None) if policy.expose_threat_model else None),
        additional_information=(
            getattr(attack, "additional_information", None) if policy.expose_additional_information else None
        ),
    )
