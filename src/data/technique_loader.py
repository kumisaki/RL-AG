"""Load mappings from policies to MITRE ATT&CK techniques."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping

from .models import PolicyKey, TechniqueMapping


class TechniqueRepository:
    """Repository for policy-to-technique mappings."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._tech_dir = data_root / "tech_dic"
        self._mappings: Dict[PolicyKey, TechniqueMapping] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for path in sorted(self._tech_dir.glob("stage*_tech.json")):
            self._load_stage_file(path)
        self._loaded = True

    def _load_stage_file(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            payload: Mapping[str, List[str]] = json.load(handle)
        for raw_key, technique_ids in payload.items():
            key = PolicyKey.parse(raw_key)
            mapping = TechniqueMapping(key=key, technique_ids=tuple(technique_ids))
            self._mappings[key] = mapping

    def iter_mappings(self) -> Iterator[TechniqueMapping]:
        self.load()
        return iter(self._mappings.values())

    def techniques_for_policy(self, key: PolicyKey) -> TechniqueMapping:
        self.load()
        return self._mappings[key]

    def by_stage(self, stage: int) -> List[TechniqueMapping]:
        self.load()
        return [mapping for mapping in self._mappings.values() if mapping.key.stage == stage]

    def by_tactic(self, tactic: str) -> List[TechniqueMapping]:
        self.load()
        tactic_lower = tactic.lower()
        return [
            mapping
            for mapping in self._mappings.values()
            if mapping.key.tactic.lower() == tactic_lower
        ]

