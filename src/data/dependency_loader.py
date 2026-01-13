"""Loader for tactic dependency maps derived from TAGAPT guidance."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


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
        self._mandatory_by_parent: Dict[str, Dict[str, str]] = {}
        self._optional_by_parent: Dict[str, Dict[str, str]] = {}
        self._parent_stage_lookup: Dict[Tuple[str, str], TacticConfig] = {}
        self._parent_order: Tuple[str, ...] = tuple(
            dict.fromkeys(
                [
                    str(config.get("parent"))
                    for _, config in sorted(
                        tactics_payload.items(), key=lambda item: self._parse_tactic_key(item[0])[0]
                    )
                    if config.get("parent") is not None
                ]
            )
        )
        critical_raw = payload.get("critical_tactics")
        if critical_raw is None:
            critical_raw = payload.get("key_tactics", ["Execution", "Command and Control", "Impact"])
        movement_raw = payload.get("movement_tactics", ["Lateral Movement"])
        self._critical_tactics = {self._normalize_name(str(name)) for name in critical_raw}
        self._movement_tactics = {self._normalize_name(str(name)) for name in movement_raw}

        stage_tactic_order_payload = payload.get("stage_tactic_order", {})
        self._parent_to_stages: Dict[str, Tuple[int, ...]] = {
            parent: tuple(int(value) for value in values)
            for parent, values in stage_tactic_order_payload.items()
        }

        self._name_lookup: Dict[str, TacticConfig] = {}
        for tactic_str, config in tactics_payload.items():
            tactic_id, tactic_label = self._parse_tactic_key(tactic_str)
            mandatory = tuple(self._clean_label(value) for value in (config.get("mandatory", []) or []))
            optional = tuple(self._clean_label(value) for value in (config.get("optional", []) or []))
            parent = config.get("parent")
            self._configs[tactic_id] = TacticConfig(
                tactic_id=tactic_id,
                mandatory=mandatory,
                optional=optional,
                parent_stage=str(parent) if parent is not None else None,
            )
            self._mandatory_lookup[tactic_id] = {name.lower(): name for name in mandatory}
            self._optional_lookup[tactic_id] = {name.lower(): name for name in optional}
            if parent is not None:
                parent_key = str(parent)
                mandatory_map = self._mandatory_by_parent.setdefault(parent_key, {})
                optional_map = self._optional_by_parent.setdefault(parent_key, {})
                for name in mandatory:
                    mandatory_map.setdefault(name.lower(), name)
                for name in optional:
                    optional_map.setdefault(name.lower(), name)
                self._parent_stage_lookup[(parent_key, str(tactic_id))] = self._configs[tactic_id]
                # also index by lower-cased tactic names for fallback
                for name in mandatory + optional:
                    self._parent_stage_lookup[(parent_key, name.lower())] = self._configs[tactic_id]
            for name in mandatory + optional:
                self._name_lookup.setdefault(name.lower(), self._configs[tactic_id])

        if not self._parent_to_stages:
            parent_map: Dict[str, List[int]] = {}
            for tactic_id, config in self._configs.items():
                if config.parent_stage:
                    parent_map.setdefault(config.parent_stage, []).append(tactic_id)
            self._parent_to_stages = {
                parent: tuple(sorted(values)) for parent, values in parent_map.items()
            }

    def _parse_tactic_key(self, raw_key: str) -> Tuple[int, Optional[str]]:
        if "." in raw_key:
            prefix, _, label = raw_key.partition(".")
        else:
            prefix, label = raw_key, None
        try:
            tactic_id = int(prefix)
        except ValueError:
            tactic_id = int(prefix.strip() or 0)
        return tactic_id, label

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
        return self._parent_to_stages.get(parent_stage, tuple())

    def classify_tactic_for_parent(self, parent: Optional[str], tactic_name: str) -> Optional[str]:
        if not parent:
            return None
        lookup_name = tactic_name.lower()
        if lookup_name in self._mandatory_by_parent.get(parent, {}):
            return "mandatory"
        if lookup_name in self._optional_by_parent.get(parent, {}):
            return "optional"
        return None

    def tactic_config_for_parent(self, parent: Optional[str], tactic_name: str) -> Optional[TacticConfig]:
        if not parent:
            return None
        lookup = (parent, tactic_name.lower())
        return self._parent_stage_lookup.get(lookup)
    # get the tactic category for a given stage and tactic name
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

    def critical_tactics(self) -> Tuple[str, ...]:
        return tuple(sorted(self._critical_tactics))

    def is_critical_tactic(self, tactic_name: str) -> bool:
        return tactic_name.lower() in self._critical_tactics

    def movement_tactics(self) -> Tuple[str, ...]:
        return tuple(sorted(self._movement_tactics))

    def is_movement_tactic(self, tactic_name: str) -> bool:
        return tactic_name.lower() in self._movement_tactics

    def mandatory_tactics_for_parent(self, parent: Optional[str]) -> Tuple[str, ...]:
        if not parent:
            return tuple()
        mapping = self._mandatory_by_parent.get(parent, {})
        return tuple(sorted(mapping.values()))

    def optional_tactics_for_parent(self, parent: Optional[str]) -> Tuple[str, ...]:
        if not parent:
            return tuple()
        mapping = self._optional_by_parent.get(parent, {})
        return tuple(sorted(mapping.values()))

    def config_for_tactic_name(self, tactic_name: str) -> Optional[TacticConfig]:
        return self._name_lookup.get(self._normalize_name(tactic_name))

    def _clean_label(self, name: str) -> str:
        if "." in name:
            _, _, tail = name.partition(".")
            if tail:
                return tail.strip()
        return name.strip()

    def _normalize_name(self, name: str) -> str:
        return self._clean_label(name).lower()
