"""Loader for tactic dependency maps derived from TAGAPT guidance."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class TacticConfig:
    tactic_id: int
    mandatory: Tuple[str, ...]
    optional: Tuple[str, ...]
    parent_stage: Optional[str]


class TacticDependencyMap:
    """Provides access to stage-level mandatory/optional tactics."""

    def __init__(self, data_root: Path) -> None:
        self._path = data_root / "tactic_dependency_map" / "dependencies.json"
        with self._path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        # Support legacy schema (stage-centric) and new tactic-centric schema.
        tactics_payload: Dict[str, Dict[str, object]] = {}
        tactic_order = payload.get("tactic_order")
        if tactic_order:
            self._tactic_order: Tuple[int, ...] = tuple(int(value) for value in tactic_order)
            tactics_payload = payload.get("tactics", {})
        else:
            stage_order = payload.get("stage_order", [])
            self._tactic_order = tuple(int(value) for value in stage_order)
            stages_payload = payload.get("stages", {})
            for stage_str, config in stages_payload.items():
                tactics_payload[stage_str] = {
                    "mandatory": config.get("mandatory", []),
                    "optional": config.get("optional", []),
                    "parent": config.get("parent"),
                }

        self._configs: Dict[int, TacticConfig] = {}
        self._mandatory_lookup: Dict[int, Dict[str, str]] = {}
        self._optional_lookup: Dict[int, Dict[str, str]] = {}
        self._parent_order: Tuple[str, ...] = tuple(
            dict.fromkeys(
                [
                    str(config.get("parent"))
                    for _, config in sorted(
                        tactics_payload.items(), key=lambda item: int(item[0])
                    )
                    if config.get("parent") is not None
                ]
            )
        )

        for tactic_str, config in tactics_payload.items():
            tactic_id = int(tactic_str)
            mandatory = tuple(config.get("mandatory", []) or [])
            optional = tuple(config.get("optional", []) or [])
            parent = config.get("parent")
            self._configs[tactic_id] = TacticConfig(
                tactic_id=tactic_id,
                mandatory=mandatory,
                optional=optional,
                parent_stage=str(parent) if parent is not None else None,
            )
            self._mandatory_lookup[tactic_id] = {name.lower(): name for name in mandatory}
            self._optional_lookup[tactic_id] = {name.lower(): name for name in optional}

    def tactic_order(self) -> Tuple[int, ...]:
        return self._tactic_order

    def stage_order(self) -> Tuple[int, ...]:
        """Alias for backward compatibility with legacy stage semantics."""
        return self._tactic_order

    def parent_stage_order(self) -> Tuple[str, ...]:
        return self._parent_order

    def mandatory_tactics(self, stage: int) -> Tuple[str, ...]:
        config = self._configs.get(stage)
        if not config:
            return tuple()
        return config.mandatory

    def optional_tactics(self, stage: int) -> Tuple[str, ...]:
        config = self._configs.get(stage)
        if not config:
            return tuple()
        return config.optional

    def parent_stage(self, stage: int) -> Optional[str]:
        config = self._configs.get(stage)
        return config.parent_stage if config else None

    def tactic_config(self, stage: int) -> Optional[TacticConfig]:
        return self._configs.get(stage)

    def tactics_for_stage(self, parent_stage: str) -> Tuple[int, ...]:
        return tuple(
            tactic_id
            for tactic_id in self._tactic_order
            if self._configs.get(tactic_id, TacticConfig(tactic_id, tuple(), tuple(), None)).parent_stage
            == parent_stage
        )

    def classify_tactic(self, stage: int, tactic_name: str) -> Optional[str]:
        """Return 'mandatory', 'optional', or None if not part of this stage entry."""
        config = self._configs.get(stage)
        if not config:
            return None
        lookup_name = tactic_name.lower()
        if lookup_name in self._mandatory_lookup.get(stage, {}):
            return "mandatory"
        if lookup_name in self._optional_lookup.get(stage, {}):
            return "optional"
        return None

    def iter_configs(self) -> Iterable[TacticConfig]:
        for tactic_id in self._tactic_order:
            config = self._configs.get(tactic_id)
            if config:
                yield config
