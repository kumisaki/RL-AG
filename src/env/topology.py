"""Static IIoT topology representation consumed by the RL environment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class VulnerabilityRecord:
    cve_id: str
    cvss_base: float
    epss: float
    mapped_techniques: Tuple[str, ...]


@dataclass
class DeviceConnection:
    target: str
    segment: str
    protocols: Tuple[str, ...]
    direction: str
    notes: Optional[str] = None


@dataclass
class DeviceNode:
    device_id: str
    name: str
    device_type: str
    ip_address: Optional[str]
    platform: Optional[str]
    neighbors: Tuple[str, ...]
    connections: Tuple[DeviceConnection, ...]
    vulnerabilities: Tuple[VulnerabilityRecord, ...] = ()

    @property
    def mapped_techniques(self) -> Set[str]:
        techniques: Set[str] = set()
        for record in self.vulnerabilities:
            techniques.update(record.mapped_techniques)
        return techniques


class TopologyGraph:
    """Adapts the JSON topology description to a graph abstraction."""

    def __init__(self, nodes: Mapping[str, DeviceNode]) -> None:
        self._nodes: Dict[str, DeviceNode] = dict(nodes)

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "TopologyGraph":
        nodes = {
            node_dict["device_id"]: _parse_device_node(node_dict)
            for node_dict in data.get("nodes", [])
        }
        return cls(nodes)

    @classmethod
    def from_file(cls, path: Path) -> "TopologyGraph":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_json(payload)

    def node_ids(self) -> Tuple[str, ...]:
        return tuple(self._nodes.keys())

    def get_node(self, device_id: str) -> DeviceNode:
        return self._nodes[device_id]

    def neighbors(self, device_id: str) -> Tuple[str, ...]:
        return self._nodes[device_id].neighbors

    def devices_supporting(self, technique_id: str) -> Tuple[DeviceNode, ...]:
        technique_norm = technique_id.upper()
        return tuple(
            node for node in self._nodes.values() if technique_norm in node.mapped_techniques
        )

    def __len__(self) -> int:
        return len(self._nodes)


def _parse_device_node(node_dict: Mapping[str, object]) -> DeviceNode:
    connections_data: Sequence[Mapping[str, object]] = node_dict.get("connections", [])
    vulnerabilities_data: Sequence[Mapping[str, object]] = node_dict.get("vulnerabilities", [])
    return DeviceNode(
        device_id=node_dict["device_id"],
        name=node_dict.get("name", node_dict["device_id"]),
        device_type=node_dict.get("device_type", "Unknown"),
        ip_address=node_dict.get("ip_address"),
        platform=node_dict.get("platform"),
        neighbors=tuple(node_dict.get("neighbors", [])),
        connections=tuple(_parse_connection(c) for c in connections_data),
        vulnerabilities=tuple(_parse_vulnerability(v) for v in vulnerabilities_data),
    )


def _parse_connection(conn_dict: Mapping[str, object]) -> DeviceConnection:
    return DeviceConnection(
        target=conn_dict["target"],
        segment=conn_dict.get("segment", ""),
        protocols=tuple(conn_dict.get("protocols", [])),
        direction=conn_dict.get("direction", "bidirectional"),
        notes=conn_dict.get("notes"),
    )


def _parse_vulnerability(vuln_dict: Mapping[str, object]) -> VulnerabilityRecord:
    return VulnerabilityRecord(
        cve_id=vuln_dict["cve_id"],
        cvss_base=float(vuln_dict.get("cvss_base", 0.0)),
        epss=float(vuln_dict.get("epss", 0.0)),
        mapped_techniques=tuple(vuln_dict.get("mapped_techniques", [])),
    )
