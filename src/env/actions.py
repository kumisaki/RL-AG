"""Macro-action abstraction bridging TAGAPT policies to RL actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from data.dependency_loader import TacticDependencyMap
from data.models import PolicyDefinition, PolicyKey, TechniqueMapping
from .topology import TopologyGraph


@dataclass(frozen=True)
class AttackMacroAction:
    index: int
    stage: int
    tactic: str
    policy_key: Optional[PolicyKey]
    technique_id: str
    target_device: str
    tactic_category: str = "neutral"
    traits: Tuple[str, ...] = ()
    supports_technique: bool = True
    kind: str = "policy"


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
        self._support_exempt_stages = {1, 2}
        self._support_exempt_tactics = {"reconnaissance", "resource-development"}
        self._build_action_catalog()

    def _build_action_catalog(self) -> None:
        cursor = 0
        for stage in sorted({key.stage for key in self._policy_by_key}):
            stage_actions: List[AttackMacroAction] = []
            parent_stage = self._dependencies.parent_stage(stage) if self._dependencies else None
            stage_mandatory: Set[str] = set()
            stage_optional: Set[str] = set()
            parent_mandatory: Set[str] = set()
            parent_optional: Set[str] = set()
            if self._dependencies:
                stage_mandatory = {
                    name.lower() for name in self._dependencies.mandatory_tactics(stage)
                }
                stage_optional = {
                    name.lower() for name in self._dependencies.optional_tactics(stage)
                }
                if parent_stage:
                    parent_mandatory = {
                        name.lower()
                        for name in self._dependencies.mandatory_tactics_for_parent(parent_stage)
                    }
                    parent_optional = {
                        name.lower()
                        for name in self._dependencies.optional_tactics_for_parent(parent_stage)
                    }
            optional_only_stage = bool(self._dependencies) and not stage_mandatory
            for policy in self._stage_policies(stage):
                mapping = self._tech_by_key.get(policy.key)
                if not mapping:
                    continue
                technique_ids = mapping.technique_ids
                if self._techniques_per_policy:
                    technique_ids = technique_ids[: self._techniques_per_policy]
                for technique_id in technique_ids:
                    supported_devices = self._topology.devices_supporting(technique_id)
                    supported_ids = [device.device_id for device in supported_devices]
                    all_nodes = list(self._topology.node_ids())
                    tactic_lower = policy.key.tactic.lower()
                    support_required = (
                        stage not in self._support_exempt_stages
                        and tactic_lower not in self._support_exempt_tactics
                    )
                    targets = supported_ids if support_required else all_nodes
                    if self._targets_per_technique is not None:
                        targets = targets[: self._targets_per_technique]
                    trait_set: set[str] = set()
                    if self._dependencies:
                        if self._dependencies.is_critical_tactic(policy.key.tactic):
                            trait_set.add("critical")
                        if self._dependencies.is_movement_tactic(policy.key.tactic):
                            trait_set.add("movement")
                    if support_required and not supported_ids:
                        continue
                    for device_id in targets:
                        category = "neutral"
                        if self._dependencies:
                            normalized = policy.key.tactic.lower()
                            cls = self._dependencies.classify_tactic_for_parent(
                                parent_stage, policy.key.tactic
                            )
                            if not cls:
                                cls = self._dependencies.classify_tactic(stage, policy.key.tactic)
                            if cls:
                                category = cls
                            else:
                                if normalized in parent_mandatory or normalized in stage_mandatory:
                                    category = "mandatory"
                                elif normalized in parent_optional or normalized in stage_optional:
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
                                traits=tuple(sorted(trait_set)),
                                supports_technique=(
                                    True
                                    if not support_required
                                    else bool(supported_ids) and device_id in supported_ids
                                ),
                            )
                        )
            if optional_only_stage:
                stage_actions.append(
                    AttackMacroAction(
                        index=cursor + len(stage_actions),
                        stage=stage,
                        tactic=f"Skip Optional Stage {stage}",
                        policy_key=None,
                        technique_id="__SKIP__",
                        target_device="*",
                        tactic_category="optional",
                        traits=("skip",),
                        supports_technique=True,
                        kind="skip",
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
