"""Gym-style environment stitching together topology, policies, and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Set, List, Sequence

from data.policy_loader import PolicyRepository
from data.technique_loader import TechniqueRepository
from data.instance_library import TechniqueInstanceLibrary
from data.models import PolicyDefinition, PolicyKey, ProvenanceTriple
from data.dependency_loader import TacticDependencyMap

from .actions import AttackMacroAction, MacroActionSpace, ActionMask
from .domain import DomainConstraintEngine, DomainRuleViolation
from .provenance import ProvenanceState, ProvenanceNode
from .topology import TopologyGraph

try:  # pragma: no cover - optional dependency
    import gymnasium as gym  # type: ignore
except ImportError:  # pragma: no cover
    try:
        import gym  # type: ignore
    except ImportError:  # pragma: no cover
        gym = None  # type: ignore


@dataclass
class AttackObservation:
    topology: TopologyGraph
    provenance: ProvenanceState
    stage: int
    step_count: int
    action_mask: ActionMask
    reward_breakdown: Dict[str, float] = field(default_factory=dict)


class InstanceSampler:
    """Deterministic sampler cycling through available technique instances."""

    def __init__(self) -> None:
        self._cursor: Dict[Tuple[str, str], int] = {}

    def sample(self, technique_id: str, entity_type: str, candidates: Tuple[str, ...]) -> Optional[str]:
        if not candidates:
            return None
        key = (technique_id, entity_type)
        idx = self._cursor.get(key, 0)
        value = candidates[idx % len(candidates)]
        self._cursor[key] = idx + 1
        return value


class APTAttackEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Gym-compatible environment orchestrating TAGAPT-derived policies."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology: TopologyGraph,
        policy_repo: PolicyRepository,
        technique_repo: TechniqueRepository,
        instance_library: TechniqueInstanceLibrary,
        stage_order: Optional[Tuple[int, ...]] = None,
        max_steps: Optional[int] = None,
        dependency_map: Optional[TacticDependencyMap] = None,
        stage_patience: int = 8,
        stage_completion_bonus: float = 5.0,
        stage_transition_bonus: float = 1.0,
        techniques_per_policy: Optional[int] = 3,
        targets_per_technique: Optional[int] = 5,
    ) -> None:
        self._topology = topology
        self._policy_repo = policy_repo
        self._tech_repo = technique_repo
        self._instance_library = instance_library
        policy_stages = sorted({policy.key.stage for policy in policy_repo.iter_policies()})
        if dependency_map and dependency_map.tactic_order():
            ordered = [stage for stage in dependency_map.tactic_order() if stage in policy_stages]
        else:
            ordered = list(policy_stages)
        if not ordered:
            raise ValueError("No tactics available from policy repository.")
        self._all_tactic_ids = tuple(ordered)
        if stage_order:
            filtered = [stage for stage in stage_order if stage in self._all_tactic_ids]
            if not filtered:
                raise ValueError("Provided stage_order does not intersect available tactics.")
            self._active_tactic_ids = tuple(dict.fromkeys(filtered))
        else:
            self._active_tactic_ids = self._all_tactic_ids
        self._tactic_index_map = {stage: idx for idx, stage in enumerate(self._active_tactic_ids)}
        self._max_steps = max_steps if max_steps is not None else 200
        self._dependencies = dependency_map
        policies = list(policy_repo.iter_policies())
        self._policies = policies
        self._parent_lookup: Dict[int, Optional[str]] = {}
        if self._dependencies:
            for stage in self._all_tactic_ids:
                self._parent_lookup[stage] = self._dependencies.parent_stage(stage)
        else:
            for stage in self._all_tactic_ids:
                self._parent_lookup[stage] = f"stage_{stage}"
        self._parent_by_stage = {stage: self._parent_lookup.get(stage) for stage in self._active_tactic_ids}
        self._policy_parent: Dict[PolicyKey, str] = {}
        self._stage_parent_map: Dict[int, str] = {}
        self._tactic_classification: Dict[str, str] = {}
        self._build_policy_metadata(policies)
        self._optional_only_stages: Set[int] = set()
        self._rebuild_target_sets()
        self._stage_patience = max(stage_patience, 1)
        self._stage_completion_bonus = stage_completion_bonus
        self._stage_transition_bonus = stage_transition_bonus
        self._mandatory_tactic_reward = max(stage_transition_bonus, 1.0)
        self._optional_tactic_reward = self._mandatory_tactic_reward * 0.5
        self._impact_bonus = max(2 * self._stage_completion_bonus + 1.0, 11.0)
        techniques = list(technique_repo.iter_mappings())
        self._action_space_helper = MacroActionSpace(
            self._topology,
            policies,
            techniques,
            dependency_map=self._dependencies,
            techniques_per_policy=techniques_per_policy,
            targets_per_technique=targets_per_technique,
        )
        self._domain_engine = DomainConstraintEngine(policies)
        self._sampler = InstanceSampler()
        self._provenance = ProvenanceState()
        self._step_count = 0
        self._stage_step_counter = 0
        self._stage_visit_counts: Dict[int, int] = {}
        self._pending_stage_completion: Dict[int, int] = {}
        self._mandatory_progress: Dict[str, Set[str]] = {}
        self._optional_progress: Dict[str, Set[str]] = {}
        self._completed_stages: Set[int] = set()
        self._available_indices: Set[int] = set()
        self._unlocked_parents: Set[str] = set()
        self._campaign_final_stage = self._active_tactic_ids[-1]
        self._last_progress_step = 0
        self._stagnation_penalty_rate = 0.01
        self._stagnation_penalty_cap = 0.5
        self._initialize_progress_trackers()
        
        # Network topology tracking for reachability
        self._compromised_devices: Set[str] = set()
        self._initial_entry_device: Optional[str] = None
        self._previously_compromised: Set[str] = set()

        self.action_space = gym.spaces.Discrete(len(self._action_space_helper.all_actions())) if gym else None
        self.observation_space = None  # handled by custom dataclass

    def action_count(self) -> int:
        """Total number of macro-actions exposed to the agent."""
        return len(self._action_space_helper.all_actions())

    def _initialize_progress_trackers(self) -> None:
        self._stage_visit_counts = {stage: 0 for stage in self._active_tactic_ids}
        self._pending_stage_completion = {}
        parent_ids = set(self._stage_parent_map.get(stage) or "" for stage in self._active_tactic_ids)
        self._mandatory_progress = {parent: set() for parent in parent_ids}
        self._optional_progress = {parent: set() for parent in parent_ids}
        self._completed_stages = set()
        self._available_indices = set()
        if self._active_tactic_ids:
            self._available_indices.add(0)
            first_parent = self._parent_by_stage.get(self._active_tactic_ids[0])
            self._unlocked_parents = set()
            if first_parent:
                self._unlocked_parents.add(first_parent)
            self._auto_complete_if_no_mandatory(self._active_tactic_ids[0], reward_breakdown=None)
        else:
            self._unlocked_parents = set()
        self._last_progress_step = 0
        self._stage_step_counter = 0

    def set_stage_order(self, stages: Tuple[int, ...]) -> None:
        """Override progression order, used by curriculum scheduling."""
        if not stages:
            raise ValueError("Stage order cannot be empty")
        unique = []
        for stage in stages:
            if stage not in unique and stage in self._all_tactic_ids:
                unique.append(stage)
        if not unique:
            raise ValueError("Stage order must reference known tactics.")
        ordered = tuple(sorted(unique, key=lambda s: self._all_tactic_ids.index(s)))
        self._active_tactic_ids = ordered
        self._tactic_index_map = {stage: idx for idx, stage in enumerate(self._active_tactic_ids)}
        self._parent_by_stage = {stage: self._parent_lookup.get(stage) for stage in self._active_tactic_ids}
        self._rebuild_target_sets()
        self._campaign_final_stage = self._active_tactic_ids[-1]
        self._initialize_progress_trackers()

    def _rebuild_target_sets(self) -> None:
        self._mandatory_targets: Dict[str, Set[str]] = {}
        self._optional_targets: Dict[str, Set[str]] = {}
        if self._dependencies:
            self._critical_tactics = {name.lower() for name in self._dependencies.critical_tactics()}
            self._movement_tactics = {name.lower() for name in self._dependencies.movement_tactics()}
        else:
            self._critical_tactics = {"execution", "command and control", "impact"}
            self._movement_tactics = {"lateral movement"}
        for policy in self._policies:
            parent = self._policy_parent.get(policy.key)
            if not parent:
                continue
            normalized = policy.key.tactic.lower()
            category = self._tactic_classification.get(normalized, "neutral")
            if category == "mandatory":
                self._mandatory_targets.setdefault(parent, set()).add(normalized)
            elif category == "optional":
                self._optional_targets.setdefault(parent, set()).add(normalized)
        self._impact_parent_ids = {
            parent for parent, tactics in self._mandatory_targets.items() if "impact" in tactics
        }
        self._impact_indices = {
            self._tactic_index_map[stage]
            for stage in self._active_tactic_ids
            if self._stage_parent_map.get(stage) in self._impact_parent_ids
        }
        self._refresh_optional_stage_flags()

    def _refresh_optional_stage_flags(self) -> None:
        self._optional_only_stages = set()
        for stage in self._active_tactic_ids:
            parent = self._stage_parent_map.get(stage)
            has_mandatory = bool(self._mandatory_targets.get(parent or "", set()))
            if not has_mandatory:
                self._optional_only_stages.add(stage)

    def _device_platform(self, device_id: str) -> Optional[str]:
        """Best-effort lookup for the platform/OS associated with a device."""
        try:
            node = self._topology.get_node(device_id)
        except KeyError:
            return None
        return getattr(node, "platform", None)

    def _stage_reward_multiplier(self, stage: int) -> float:
        if not self._active_tactic_ids:
            return 1.0
        index = self._tactic_index_map.get(stage, 0)
        denom = max(1, len(self._active_tactic_ids) - 1)
        return 1.0 + index / denom

    def _build_policy_metadata(self, policies: Sequence[PolicyDefinition]) -> None:
        classification: Dict[str, str] = {}
        tactic_parent: Dict[str, str] = {}
        if self._dependencies:
            for config in self._dependencies.iter_configs():
                base_parent = config.parent_stage or f"tactic_{config.tactic_id}"
                parent = f"{base_parent}#{config.tactic_id}"
                for name in config.mandatory:
                    normalized = name.lower()
                    classification[normalized] = "mandatory"
                    tactic_parent.setdefault(normalized, parent)
                for name in config.optional:
                    normalized = name.lower()
                    classification.setdefault(normalized, "optional")
                    tactic_parent.setdefault(normalized, parent)
        for policy in policies:
            normalized = policy.key.tactic.lower()
            parent = tactic_parent.get(normalized)
            if not parent and self._dependencies:
                config = self._dependencies.config_for_tactic_name(normalized)
                if config:
                    base_parent = config.parent_stage or f"tactic_{config.tactic_id}"
                    parent = f"{base_parent}#{config.tactic_id}"
                    if normalized in {name.lower() for name in config.mandatory}:
                        classification[normalized] = "mandatory"
                    elif normalized in {name.lower() for name in config.optional}:
                        classification.setdefault(normalized, "optional")
            if not parent:
                parent = policy.stage_group or f"stage_{policy.key.stage}"
            self._policy_parent[policy.key] = parent
            self._stage_parent_map.setdefault(policy.key.stage, parent)
            classification.setdefault(normalized, "neutral")
        self._tactic_classification = classification

    def _classify_tactic(self, tactic_name: str) -> str:
        return self._tactic_classification.get(tactic_name.lower(), "neutral")

    def provenance_state(self) -> ProvenanceState:
        """Return the underlying provenance graph (read-only usage expected)."""
        return self._provenance
    
    def _select_entry_point(self) -> str:
        """Select initial entry point for attack (typically internet-facing device)."""
        # Prefer workstations, servers, or engineering stations as entry points
        preferred_types = ["Workstation", "Server", "EngineeringWorkstation", "HMI"]
        for device_id in self._topology.node_ids():
            device = self._topology.get_node(device_id)
            if device.device_type in preferred_types:
                return device_id
        # Fallback: return first device
        return self._topology.node_ids()[0]
    
    def _is_lateral_movement_technique(self, technique_id: str) -> bool:
        """Check if technique is for lateral movement."""
        # Check technique mappings to find the tactic
        for mapping in self._tech_repo.iter_mappings():
            if technique_id in mapping.technique_ids:
                tactic_lower = mapping.key.tactic.lower()
                return tactic_lower in self._movement_tactics
        return False
    
    def _is_device_reachable(self, target_device: str, technique_id: str) -> bool:
        """Check if target device is reachable from compromised devices."""
        # If no devices compromised yet, allow initial access
        if not self._compromised_devices:
            return True
        
        # Target already compromised - always reachable for persistence/privilege escalation
        if target_device in self._compromised_devices:
            return True
        
        # Check if this is a lateral movement technique
        is_lateral_movement = self._is_lateral_movement_technique(technique_id)
        
        # Lateral movement techniques can reach any device
        if is_lateral_movement:
            return True
        
        # Otherwise, target must be a neighbor of a compromised device
        for compromised in self._compromised_devices:
            neighbors = self._topology.neighbors(compromised)
            if target_device in neighbors:
                return True
        
        # Not reachable
        return False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        del seed, options
        self._domain_engine.reset()
        self._provenance = ProvenanceState()
        self._step_count = 0
        self._initialize_progress_trackers()
        
        # Reset network topology tracking
        self._compromised_devices.clear()
        self._previously_compromised.clear()
        self._initial_entry_device = self._select_entry_point()
        self._compromised_devices.add(self._initial_entry_device)
        
        observation = self._build_observation({})
        return observation, {}

    def step(self, action_index: int):  # type: ignore[override]
        action = self._action_space_helper.all_actions()[action_index]
        reward_breakdown = {"structure": 0.0, "temporal": 0.0, "utility": -0.01}

        stage_id = action.stage
        stage_idx = self._tactic_index_map.get(stage_id)
        self._step_count += 1
        truncated = self._max_steps is not None and self._step_count >= self._max_steps
        self._stage_step_counter += 1

        if stage_idx is None:
            reward_breakdown["utility"] -= 0.5
            observation = self._build_observation(reward_breakdown)
            return observation, sum(reward_breakdown.values()), False, truncated, {
                "reward_breakdown": reward_breakdown,
                "action": action,
                "invalid": True,
                "reason": "unknown_stage",
            }

        if stage_idx not in self._available_indices:
            reward_breakdown["utility"] -= 0.5
            observation = self._build_observation(reward_breakdown)
            return observation, sum(reward_breakdown.values()), False, truncated, {
                "reward_breakdown": reward_breakdown,
                "action": action,
                "invalid": True,
                "reason": "locked_stage",
            }

        if getattr(action, "kind", "policy") == "skip":
            traits = set(getattr(action, "traits", ()))
            if stage_id not in self._optional_only_stages:
                reward_breakdown["utility"] -= 0.5
                observation = self._build_observation(reward_breakdown)
                return observation, sum(reward_breakdown.values()), False, truncated, {
                    "reward_breakdown": reward_breakdown,
                    "action": action,
                    "invalid": True,
                    "reason": "invalid_skip",
                }
            self._stage_visit_counts[stage_id] = self._stage_visit_counts.get(stage_id, 0) + 1
            self._complete_stage(stage_id, reward_breakdown, add_pending=False)
            final_stage_complete = self._campaign_final_stage in self._completed_stages
            observation = self._build_observation(reward_breakdown)
            info = {
                "reward_breakdown": reward_breakdown,
                "action": action,
                "classification": "skip",
                "traits": tuple(sorted(traits)),
                "episode_complete": final_stage_complete,
                "skipped_optional_stage": True,
            }
            return observation, sum(reward_breakdown.values()), final_stage_complete, truncated, info

        traits = set(getattr(action, "traits", ()))
        tactic_name_lower = action.tactic.lower()
        if tactic_name_lower in self._critical_tactics:
            traits.add("critical")
        if tactic_name_lower in self._movement_tactics:
            traits.add("movement")

        classification = self._tactic_classification.get(
            tactic_name_lower, action.tactic_category or "neutral"
        )
        parent_stage = self._stage_parent_map.get(stage_id)

        if not getattr(action, "supports_technique", True):
            reward_breakdown["utility"] -= 0.5
            observation = self._build_observation(reward_breakdown)
            return observation, sum(reward_breakdown.values()), False, truncated, {
                "reward_breakdown": reward_breakdown,
                "action": action,
                "invalid": True,
                "reason": "unsupported_target",
            }

        if "critical" in traits and tactic_name_lower != "impact":
            self._unlock_impact_stage(reward_breakdown)

        self._stage_visit_counts[stage_id] = self._stage_visit_counts.get(stage_id, 0) + 1

        policy = self._policy_repo.by_key(action.policy_key)
        try:
            self._apply_policy(policy, action, reward_breakdown)
        except DomainRuleViolation as exc:
            reward_breakdown["structure"] -= 0.5
            observation = self._build_observation(reward_breakdown)
            return observation, sum(reward_breakdown.values()), False, truncated, {
                "reward_breakdown": reward_breakdown,
                "action": action,
                "invalid": True,
                "error": str(exc),
            }
        
        # Mark target device as compromised
        if action.target_device not in self._previously_compromised:
            self._previously_compromised.add(action.target_device)
            reward_breakdown["utility"] += 0.2  # Small bonus for expanding attack surface
        self._compromised_devices.add(action.target_device)

        self._register_tactic_completion(
            stage_id,
            action.tactic,
            classification,
            reward_breakdown,
            traits,
            action.target_device,
        )

        if not truncated:
            stagnation_penalty = self._stage_stagnation_penalty()
            if stagnation_penalty:
                reward_breakdown["temporal"] -= stagnation_penalty

        final_stage_complete = self._campaign_final_stage in self._completed_stages
        observation = self._build_observation(reward_breakdown)
        reward = sum(reward_breakdown.values())
        info = {
            "reward_breakdown": reward_breakdown,
            "action": action,
            "classification": classification,
            "traits": tuple(sorted(traits)),
            "episode_complete": final_stage_complete,
        }
        return observation, reward, final_stage_complete, truncated, info

    def available_actions(self) -> Tuple[AttackMacroAction, ...]:
        stages = self._unlocked_stages()
        actions: List[AttackMacroAction] = []
        for stage in stages:
            actions.extend(self._action_space_helper.actions_for_stage(stage))
        return tuple(actions)

    def _apply_policy(
        self,
        policy: PolicyDefinition,
        action: AttackMacroAction,
        reward_breakdown: Dict[str, float],
    ) -> None:
        platform = self._device_platform(action.target_device)
        technique_instances = self._instance_library.instances_for(
            action.technique_id, platform=platform
        )
        made_progress = False
        for triple in policy.triples:
            subject_node = self._instantiate_entity(triple.subject, action, technique_instances)
            object_node = self._instantiate_entity(triple.obj, action, technique_instances)
            self._domain_engine.validate(triple, subject_node, object_node)
            self._provenance.add_edge(
                subject=subject_node,
                relation=triple.relation,
                obj=object_node,
                technique_id=action.technique_id,
                policy_key=policy.key,
                metadata={"target_device": action.target_device},
            )
            self._domain_engine.register(triple, subject_node, object_node)
            reward_breakdown["structure"] += 0.1
            made_progress = True

        if policy.key.stage == self._campaign_final_stage:
            reward_breakdown["utility"] += 1.0 * self._stage_reward_multiplier(policy.key.stage)
        if made_progress:
            self._mark_progress()

    def _register_tactic_completion(
        self,
        stage: int,
        tactic: str,
        classification: str,
        reward_breakdown: Dict[str, float],
        traits: Set[str],
        target_device: str,
    ) -> None:
        normalized = tactic.lower()
        parent = self._stage_parent_map.get(stage)
        parent_key = parent or ""
        stage_mandatory = self._mandatory_targets.get(parent_key, set())
        stage_optional = self._optional_targets.get(parent_key, set())
        effective_optional = stage_optional
        mandatory_targets = stage_mandatory
        has_mandatory = bool(mandatory_targets)
        optional_only_stage = stage in self._optional_only_stages
        is_critical = "critical" in traits or normalized in self._critical_tactics
        is_impact = normalized == "impact"
        reward_factor = self._stage_reward_multiplier(stage)
        if classification == "mandatory":
            targets = mandatory_targets
            if not targets:
                self._complete_stage(stage, reward_breakdown, add_pending=False)
                return
            progress = self._mandatory_progress.setdefault(parent_key, set())
            if normalized not in progress:
                progress.add(normalized)
                reward_breakdown["utility"] += self._mandatory_tactic_reward * reward_factor
                self._mark_progress()
            if targets.issubset(progress):
                self._complete_stage(stage, reward_breakdown, add_pending=True)
        elif classification == "optional":
            progress = self._optional_progress.setdefault(parent_key, set())
            if normalized not in progress:
                progress.add(normalized)
                reward_breakdown["utility"] += self._optional_tactic_reward * reward_factor
                self._mark_progress()
            if not has_mandatory:
                if optional_only_stage or not effective_optional:
                    self._complete_stage(stage, reward_breakdown, add_pending=True)
                elif effective_optional.issubset(progress):
                    self._complete_stage(stage, reward_breakdown, add_pending=True)
        if is_critical and not is_impact:
            self._unlock_impact_stage(reward_breakdown)
        if is_impact:
            self._complete_stage(stage, reward_breakdown, add_pending=True)

    def _complete_stage(
        self,
        stage: int,
        reward_breakdown: Optional[Dict[str, float]],
        add_pending: bool,
    ) -> None:
        if stage in self._completed_stages:
            return
        self._completed_stages.add(stage)
        stage_index = self._tactic_index_map.get(stage)
        if stage_index is not None:
            self._available_indices.discard(stage_index)
        if add_pending:
            if reward_breakdown is not None:
                reward_breakdown["structure"] += (
                    self._stage_completion_bonus * self._stage_reward_multiplier(stage)
                )
            self._pending_stage_completion[stage] = self._pending_stage_completion.get(stage, 0) + 1
        if stage == self._campaign_final_stage and reward_breakdown is not None:
            reward_breakdown["utility"] += self._impact_bonus
        self._mark_progress()
        self._advance_after_completion(stage, reward_breakdown)

    def _advance_after_completion(
        self,
        stage: int,
        reward_breakdown: Optional[Dict[str, float]],
    ) -> None:
        stage_index = self._tactic_index_map.get(stage)
        if stage_index is None:
            return
        next_index = stage_index + 1
        if next_index >= len(self._active_tactic_ids):
            return
        if next_index not in self._available_indices:
            self._available_indices.add(next_index)
            next_stage = self._active_tactic_ids[next_index]
            parent = self._parent_by_stage.get(next_stage)
            if parent and parent not in self._unlocked_parents:
                self._unlocked_parents.add(parent)
                if reward_breakdown is not None:
                    reward_breakdown["temporal"] += self._stage_transition_bonus
            self._auto_complete_if_no_mandatory(next_stage, reward_breakdown)

    def _auto_complete_if_no_mandatory(
        self,
        stage: int,
        reward_breakdown: Optional[Dict[str, float]],
    ) -> None:
        parent = self._stage_parent_map.get(stage)
        targets = self._mandatory_targets.get(parent or "", set())
        if targets:
            return
        stage_index = self._tactic_index_map.get(stage)
        if stage_index is None:
            return
        self._available_indices.add(stage_index)
        if stage not in self._optional_only_stages and stage not in self._completed_stages:
            # Legacy behavior: auto-complete only when the stage has neither stage-level
            # nor parent-level mandates and is not flagged as player-controllable optional.
            self._complete_stage(stage, reward_breakdown=None, add_pending=False)

    def _unlock_impact_stage(self, reward_breakdown: Optional[Dict[str, float]]) -> None:
        unlocked = False
        for idx in self._impact_indices:
            if idx not in self._available_indices:
                self._available_indices.add(idx)
                unlocked = True
        if unlocked and reward_breakdown is not None:
            reward_breakdown["temporal"] += self._stage_transition_bonus

    def _unlocked_stages(self) -> Tuple[int, ...]:
        return tuple(self._active_tactic_ids[idx] for idx in sorted(self._available_indices))

    def _current_focus_stage(self) -> int:
        for stage in self._active_tactic_ids:
            if stage not in self._completed_stages:
                return stage
        return self._active_tactic_ids[-1]

    def requires_mandatory_completion(self, stage: int) -> bool:
        parent = self._stage_parent_map.get(stage)
        return bool(self._mandatory_targets.get(parent or "", set()))

    def _stage_stagnation_penalty(self) -> float:
        if self._stage_step_counter <= self._stage_patience:
            return 0.0
        no_progress = self._step_count - self._last_progress_step
        if no_progress <= self._stage_patience:
            return 0.0
        over = no_progress - self._stage_patience
        penalty = self._stagnation_penalty_rate * over
        return min(self._stagnation_penalty_cap, penalty)

    def consume_stage_visitation(self) -> Dict[int, int]:
        snapshot = dict(self._stage_visit_counts)
        self._stage_visit_counts = {stage: 0 for stage in self._active_tactic_ids}
        return snapshot

    def consume_stage_completions(self) -> Dict[int, int]:
        snapshot = dict(self._pending_stage_completion)
        self._pending_stage_completion.clear()
        return snapshot

    def _instantiate_entity(
        self,
        entity_type: str,
        action: AttackMacroAction,
        technique_instances,
    ) -> ProvenanceNode:
        name = self._sampler.sample(
            action.technique_id, entity_type, technique_instances.get(entity_type)
        )
        return self._provenance.ensure_node(
            entity_type=entity_type,
            host_device=action.target_device,
            name=name,
        )

    def _mark_progress(self) -> None:
        self._last_progress_step = self._step_count
        self._stage_step_counter = 0

    def _build_observation(self, reward_breakdown: Dict[str, float]) -> AttackObservation:
        mask_stages = self._unlocked_stages()
        if len(mask_stages) == 1:
            base_mask = self._action_space_helper.mask_for_stage(mask_stages[0])
        else:
            base_mask = self._action_space_helper.mask_for_stages(mask_stages)
        
        # Apply reachability constraints
        action_mask_values = list(base_mask.values)
        for idx, action in enumerate(self._action_space_helper.all_actions()):
            if not action_mask_values[idx]:
                continue  # Already masked, skip
            
            # Check if target device is reachable
            if not self._is_device_reachable(action.target_device, action.technique_id):
                action_mask_values[idx] = False
        
        final_mask = ActionMask.from_actions(action_mask_values)
        
        return AttackObservation(
            topology=self._topology,
            provenance=self._provenance,
            stage=self._current_focus_stage(),
            step_count=self._step_count,
            action_mask=final_mask,
            reward_breakdown=dict(reward_breakdown),
        )
