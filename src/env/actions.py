"""Macro-action abstraction bridging TAGAPT policies to RL actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from data.dependency_loader import TacticDependencyMap
from data.models import PolicyDefinition, PolicyKey, TechniqueMapping
from .topology import TopologyGraph


@dataclass(frozen=True)
class AttackMacroAction:
    index: int
    stage: int
    tactic: str
    policy_key: PolicyKey
    technique_id: str
    target_device: str
    tactic_category: str = "neutral"


@dataclass
class ActionMask:
    values: Tuple[bool, ...]

    @classmethod
    def from_actions(cls, actions: Iterable[bool]) -> "ActionMask":
        return cls(tuple(actions))


class MacroActionSpace:
    """Enumerates possible macro-actions based on policies, techniques, and topology."""

    def __init__(
        self,
        topology: TopologyGraph,
        policies: Iterable[PolicyDefinition],
        techniques: Iterable[TechniqueMapping],
        dependency_map: Optional[TacticDependencyMap] = None,
        techniques_per_policy: Optional[int] = None,
        targets_per_technique: Optional[int] = None,
    ) -> None:
        self._topology = topology
        self._policy_by_key: Dict[PolicyKey, PolicyDefinition] = {p.key: p for p in policies}
        self._tech_by_key: Dict[PolicyKey, TechniqueMapping] = {t.key: t for t in techniques}
        self._actions: List[AttackMacroAction] = []
        self._stage_boundaries: Dict[int, Tuple[int, int]] = {}
        self._dependencies = dependency_map
        self._techniques_per_policy = techniques_per_policy
        self._targets_per_technique = targets_per_technique
        self._build_action_catalog()

    def _build_action_catalog(self) -> None:
        cursor = 0
        for stage in sorted({key.stage for key in self._policy_by_key}):
            stage_actions: List[AttackMacroAction] = []
            for policy in self._stage_policies(stage):
                mapping = self._tech_by_key.get(policy.key)
                if not mapping:
                    continue
                technique_ids = mapping.technique_ids
                if self._techniques_per_policy:
                    technique_ids = technique_ids[: self._techniques_per_policy]
                for technique_id in technique_ids:
                    supported_devices = self._topology.devices_supporting(technique_id)
                    targets = (
                        [device.device_id for device in supported_devices]
                        if supported_devices
                        else list(self._topology.node_ids())
                    )
                    if self._targets_per_technique is not None:
                        targets = targets[: self._targets_per_technique]
                    for device_id in targets:
                        category = "neutral"
                        if self._dependencies:
                            cls = self._dependencies.classify_tactic(stage, policy.key.tactic)
                            if cls:
                                category = cls
                            else:
                                normalized = policy.key.tactic.lower()
                                if normalized in {
                                    name.lower() for name in self._dependencies.mandatory_tactics(stage)
                                }:
                                    category = "mandatory"
                                elif normalized in {
                                    name.lower() for name in self._dependencies.optional_tactics(stage)
                                }:
                                    category = "optional"
                        stage_actions.append(
                            AttackMacroAction(
                                index=cursor + len(stage_actions),
                                stage=stage,
                                tactic=policy.key.tactic,
                                policy_key=policy.key,
                                technique_id=technique_id,
                                target_device=device_id,
                                tactic_category=category,
                            )
                        )
            start = cursor
            cursor += len(stage_actions)
            self._stage_boundaries[stage] = (start, cursor)
            self._actions.extend(stage_actions)

    def _stage_policies(self, stage: int) -> List[PolicyDefinition]:
        return sorted(
            (policy for policy in self._policy_by_key.values() if policy.key.stage == stage),
            key=lambda policy: policy.key.policy_number,
        )

    def actions_for_stage(self, stage: int) -> Tuple[AttackMacroAction, ...]:
        start, end = self._stage_boundaries.get(stage, (0, 0))
        return tuple(self._actions[start:end])

    def all_actions(self) -> Tuple[AttackMacroAction, ...]:
        return tuple(self._actions)

    def mask_for_stage(self, stage: int) -> ActionMask:
        mask = [False] * len(self._actions)
        start, end = self._stage_boundaries.get(stage, (0, 0))
        for idx in range(start, end):
            mask[idx] = True
        return ActionMask(tuple(mask))

    def mask_for_stages(self, stages: Iterable[int]) -> ActionMask:
        mask = [False] * len(self._actions)
        for stage in stages:
            start, end = self._stage_boundaries.get(stage, (0, 0))
            for idx in range(start, end):
                mask[idx] = True
        return ActionMask(tuple(mask))

    def stage_bounds(self, stage: int) -> Tuple[int, int]:
        return self._stage_boundaries.get(stage, (0, 0))
