"""Data models for TAGAPT-derived policy and instance metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PolicyKey:
    """Structured identifier for a TAGAPT policy entry."""

    stage: int
    tactic: str
    policy_number: int
    raw: str

    @staticmethod
    def parse(raw: str) -> "PolicyKey":
        """Parse keys of the form '3.Initial Access-21'."""
        try:
            stage_part, tail = raw.split(".", 1)
            tactic_part, number_part = tail.rsplit("-", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed policy key: {raw}") from exc

        stage = int(stage_part)
        tactic = tactic_part.strip()
        policy_number = int(number_part)
        return PolicyKey(stage=stage, tactic=tactic, policy_number=policy_number, raw=raw)

    def __str__(self) -> str:
        return self.raw


@dataclass(frozen=True)
class ProvenanceTriple:
    """Subject-relation-object triple for provenance edges."""

    subject: str
    relation: str
    obj: str


@dataclass(frozen=True)
class PolicyDefinition:
    """Full definition of a policy including provenance triples."""

    key: PolicyKey
    triples: Tuple[ProvenanceTriple, ...]
    stage_group: Optional[str] = None
    tactic_lower: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tactic_lower", self.key.tactic.lower())

    def ensure_domain_entities(self, valid_entities: Iterable[str]) -> None:
        """Validate that all entities are part of the recognized alphabet."""
        valid = set(valid_entities)
        missing = {
            node
            for triple in self.triples
            for node in (triple.subject, triple.obj)
            if node not in valid
        }
        if missing:
            raise ValueError(f"Policy {self.key} references unknown entities: {sorted(missing)}")


@dataclass(frozen=True)
class TechniqueMapping:
    """Mapping of a policy to one or more MITRE technique identifiers."""

    key: PolicyKey
    technique_ids: Tuple[str, ...]
    stage_group: Optional[str] = None
    tactic_lower: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tactic_lower", self.key.tactic.lower())


@dataclass
class TechniqueInstanceSet:
    """Concrete instances associated with a technique for a specific platform."""

    technique_id: str
    platform: Optional[str]
    instances: Mapping[str, Tuple[str, ...]]

    def get(self, entity_type: str) -> Tuple[str, ...]:
        return self.instances.get(entity_type, tuple())


EntityAlphabet = ("MP", "TP", "MF", "SF", "TF", "SO")

TechniqueInstanceCache = Dict[Tuple[str, Optional[str]], TechniqueInstanceSet]

