"""Load TAGAPT regulation policies describing provenance triple sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping

from .models import EntityAlphabet, PolicyDefinition, PolicyKey, ProvenanceTriple


class PolicyRepository:
    """Repository for regulation dictionaries grouped by stage."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._regulation_dir = data_root / "regulation_dic"
        self._policies: Dict[PolicyKey, PolicyDefinition] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for path in sorted(self._regulation_dir.glob("stage*_regulation.json")):
            self._load_stage_file(path)
        self._loaded = True

    def _load_stage_file(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            payload: Mapping[str, List[List[str]]] = json.load(handle)
        for raw_key, triples in payload.items():
            key = PolicyKey.parse(raw_key)
            triple_models = tuple(ProvenanceTriple(*triple) for triple in triples)
            policy = PolicyDefinition(key=key, triples=triple_models)
            policy.ensure_domain_entities(EntityAlphabet)
            self._policies[key] = policy

    def iter_policies(self) -> Iterator[PolicyDefinition]:
        self.load()
        return iter(self._policies.values())

    def by_key(self, key: PolicyKey) -> PolicyDefinition:
        self.load()
        return self._policies[key]

    def by_stage(self, stage: int) -> List[PolicyDefinition]:
        self.load()
        return [policy for policy in self._policies.values() if policy.key.stage == stage]

    def by_tactic(self, tactic: str) -> List[PolicyDefinition]:
        self.load()
        tactic_lower = tactic.lower()
        return [
            policy
            for policy in self._policies.values()
            if policy.key.tactic.lower() == tactic_lower
        ]

