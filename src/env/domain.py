"""Domain constraint enforcement derived from TAGAPT policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

from data.models import PolicyDefinition, ProvenanceTriple
from .provenance import ProvenanceNode


class DomainRuleViolation(RuntimeError):
    pass


@dataclass
class DomainConstraintEngine:
    """Validates provenance events against structural constraints."""

    policies: Iterable[PolicyDefinition]

    def __post_init__(self) -> None:
        self._allowed_triples: Set[Tuple[str, str, str]] = set()
        for policy in self.policies:
            for triple in policy.triples:
                self._allowed_triples.add((triple.subject, triple.relation, triple.obj))
        self.reset()

    def reset(self) -> None:
        self._deleted_nodes: Set[str] = set()

    def validate(
        self,
        triple: ProvenanceTriple,
        subject_node: ProvenanceNode,
        object_node: ProvenanceNode,
    ) -> None:
        self._validate_schema(triple, subject_node, object_node)
        self._validate_lifecycle(triple, subject_node, object_node)

    def _validate_schema(
        self,
        triple: ProvenanceTriple,
        subject_node: ProvenanceNode,
        object_node: ProvenanceNode,
    ) -> None:
        if (triple.subject, triple.relation, triple.obj) not in self._allowed_triples:
            raise DomainRuleViolation(
                f"Triple ({triple.subject}, {triple.relation}, {triple.obj}) not permitted"
            )
        if subject_node.entity_type != triple.subject:
            raise DomainRuleViolation(
                f"Subject node type {subject_node.entity_type} "
                f"does not match triple {triple.subject}"
            )
        if object_node.entity_type != triple.obj:
            raise DomainRuleViolation(
                f"Object node type {object_node.entity_type} "
                f"does not match triple {triple.obj}"
            )

    def _validate_lifecycle(
        self,
        triple: ProvenanceTriple,
        subject_node: ProvenanceNode,
        object_node: ProvenanceNode,
    ) -> None:
        if subject_node.node_id in self._deleted_nodes:
            raise DomainRuleViolation(f"Subject node {subject_node.node_id} already deleted")
        if object_node.node_id in self._deleted_nodes:
            raise DomainRuleViolation(f"Object node {object_node.node_id} already deleted")

    def register(self, triple: ProvenanceTriple, subject_node: ProvenanceNode, object_node: ProvenanceNode) -> None:
        """Update lifecycle bookkeeping after a valid triple is applied."""
        if triple.relation == "UK":
            self._deleted_nodes.add(object_node.node_id)

