"""Dynamic provenance graph maintained during an episode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from data.models import PolicyKey, ProvenanceTriple


@dataclass
class ProvenanceNode:
    node_id: str
    entity_type: str
    host_device: Optional[str]
    name: Optional[str] = None
    attributes: Dict[str, object] = field(default_factory=dict)

    def mark(self, **attrs: object) -> None:
        self.attributes.update(attrs)


@dataclass
class ProvenanceEdge:
    edge_id: str
    subject: str
    relation: str
    obj: str
    technique_id: Optional[str]
    policy_key: Optional[PolicyKey]
    metadata: Dict[str, object] = field(default_factory=dict)


class ProvenanceState:
    """Mutable provenance graph supporting domain rule validation."""

    def __init__(self) -> None:
        self._nodes: Dict[str, ProvenanceNode] = {}
        self._edges: List[ProvenanceEdge] = []
        self._node_counter = 0
        self._edge_counter = 0

    def snapshot(self) -> Tuple[Tuple[ProvenanceNode, ...], Tuple[ProvenanceEdge, ...]]:
        return tuple(self._nodes.values()), tuple(self._edges)

    def iter_nodes(self) -> Iterator[ProvenanceNode]:
        return iter(self._nodes.values())

    def iter_edges(self) -> Iterator[ProvenanceEdge]:
        return iter(self._edges)

    def get_node(self, node_id: str) -> ProvenanceNode:
        return self._nodes[node_id]

    def add_node(
        self,
        entity_type: str,
        host_device: Optional[str],
        name: Optional[str] = None,
        attributes: Optional[Mapping[str, object]] = None,
    ) -> ProvenanceNode:
        node_id = f"n{self._node_counter}"
        self._node_counter += 1
        node = ProvenanceNode(
            node_id=node_id,
            entity_type=entity_type,
            host_device=host_device,
            name=name,
            attributes=dict(attributes or {}),
        )
        self._nodes[node_id] = node
        return node

    def add_edge(
        self,
        subject: ProvenanceNode,
        relation: str,
        obj: ProvenanceNode,
        technique_id: Optional[str],
        policy_key: Optional[PolicyKey],
        metadata: Optional[Mapping[str, object]] = None,
    ) -> ProvenanceEdge:
        edge_id = f"e{self._edge_counter}"
        self._edge_counter += 1
        edge = ProvenanceEdge(
            edge_id=edge_id,
            subject=subject.node_id,
            relation=relation,
            obj=obj.node_id,
            technique_id=technique_id,
            policy_key=policy_key,
            metadata=dict(metadata or {}),
        )
        self._edges.append(edge)
        return edge

    def ensure_node(
        self,
        entity_type: str,
        host_device: Optional[str],
        name: Optional[str],
    ) -> ProvenanceNode:
        """Return an existing matching node or create a new one."""
        for node in self._nodes.values():
            if (
                node.entity_type == entity_type
                and node.host_device == host_device
                and node.name == name
            ):
                return node
        return self.add_node(entity_type=entity_type, host_device=host_device, name=name)

