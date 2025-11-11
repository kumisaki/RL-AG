"""Load concrete entity instances for ATT&CK techniques."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Optional, Tuple

from .models import EntityAlphabet, TechniqueInstanceCache, TechniqueInstanceSet


class TechniqueInstanceLibrary:
    """Provides lookup of concrete process/file/socket names per technique."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        self._instance_file = data_root / "instance_lib" / "technique-instance-lib-os-filter-add.json"
        self._cache: TechniqueInstanceCache = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        aggregate: DefaultDict[str, Dict[str, set[str]]] = defaultdict(
            lambda: {entity: set() for entity in EntityAlphabet}
        )

        with self._instance_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                raw_key = entry.pop("stage-key")
                technique_id, platform = self._split_stage_key(raw_key)
                normalized = {
                    entity: tuple(dict.fromkeys(entry.get(entity, [])))
                    for entity in EntityAlphabet
                }
                cache_key = (technique_id, platform)
                self._cache[cache_key] = TechniqueInstanceSet(
                    technique_id=technique_id,
                    platform=platform,
                    instances=normalized,
                )
                for entity in EntityAlphabet:
                    aggregate[technique_id][entity].update(normalized[entity])

        # Build platform-agnostic fallbacks
        for technique_id, entity_map in aggregate.items():
            self._cache[(technique_id, None)] = TechniqueInstanceSet(
                technique_id=technique_id,
                platform=None,
                instances={entity: tuple(sorted(values)) for entity, values in entity_map.items()},
            )

        self._loaded = True

    def _split_stage_key(self, raw_key: str) -> Tuple[str, Optional[str]]:
        technique_id, sep, platform = raw_key.partition("-")
        technique_norm = technique_id.upper()
        platform_norm = platform.lower() if sep else None
        return technique_norm, platform_norm

    def instances_for(self, technique_id: str, platform: Optional[str] = None) -> TechniqueInstanceSet:
        """Return instances for a technique, falling back to mixed-platform data if needed."""
        self.load()
        technique_norm = technique_id.upper()
        platform_norm = platform.lower() if platform else None
        key = (technique_norm, platform_norm)
        if key in self._cache:
            return self._cache[key]
        fallback_key = (technique_norm, None)
        if fallback_key in self._cache:
            return self._cache[fallback_key]
        raise KeyError(f"No instance data for technique {technique_id} platform {platform}")

    def available_platforms(self, technique_id: str) -> Tuple[str, ...]:
        """List platforms with explicit entries for a technique."""
        self.load()
        technique_norm = technique_id.upper()
        platforms = sorted(
            {
                platform
                for tech, platform in self._cache.keys()
                if tech == technique_norm and platform is not None
            }
        )
        return tuple(platforms)

