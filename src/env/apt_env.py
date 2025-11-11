"""Gym-style environment stitching together topology, policies, and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Set, List

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
        self._parent_lookup: Dict[int, Optional[str]] = {}
        if self._dependencies:
            for stage in self._all_tactic_ids:
                self._parent_lookup[stage] = self._dependencies.parent_stage(stage)
        else:
            for stage in self._all_tactic_ids:
                self._parent_lookup[stage] = f"stage_{stage}"
        self._parent_by_stage = {stage: self._parent_lookup.get(stage) for stage in self._active_tactic_ids}
        self._mandatory_targets = {
            stage: {name.lower() for name in (self._dependencies.mandatory_tactics(stage) if self._dependencies else [])}
            for stage in self._active_tactic_ids
        }
        self._optional_targets = {
            stage: {name.lower() for name in (self._dependencies.optional_tactics(stage) if self._dependencies else [])}
            for stage in self._active_tactic_ids
        }
        self._stage_patience = max(stage_patience, 1)
        self._stage_completion_bonus = stage_completion_bonus
        self._stage_transition_bonus = stage_transition_bonus
        self._mandatory_tactic_reward = max(stage_transition_bonus, 1.0)
        self._optional_tactic_reward = self._mandatory_tactic_reward * 0.5
        policies = list(policy_repo.iter_policies())
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
        self._mandatory_progress: Dict[int, Set[str]] = {}
        self._optional_progress: Dict[int, Set[str]] = {}
        self._completed_stages: Set[int] = set()
        self._available_indices: Set[int] = set()
        self._unlocked_parents: Set[str] = set()
        self._campaign_final_stage = self._active_tactic_ids[-1]
        self._last_progress_step = 0
        self._stagnation_penalty_rate = 0.02
        self._stagnation_penalty_cap = 1.0
        self._initialize_progress_trackers()

        self.action_space = gym.spaces.Discrete(len(self._action_space_helper.all_actions())) if gym else None
        self.observation_space = None  # handled by custom dataclass

    def action_count(self) -> int:
        """Total number of macro-actions exposed to the agent."""
        return len(self._action_space_helper.all_actions())

    def _initialize_progress_trackers(self) -> None:
        self._stage_visit_counts = {stage: 0 for stage in self._active_tactic_ids}
        self._pending_stage_completion = {}
        self._mandatory_progress = {stage: set() for stage in self._active_tactic_ids}
        self._optional_progress = {stage: set() for stage in self._active_tactic_ids}
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
        self._mandatory_targets = {
            stage: {name.lower() for name in (self._dependencies.mandatory_tactics(stage) if self._dependencies else [])}
            for stage in self._active_tactic_ids
        }
        self._optional_targets = {
            stage: {name.lower() for name in (self._dependencies.optional_tactics(stage) if self._dependencies else [])}
            for stage in self._active_tactic_ids
        }
        self._campaign_final_stage = self._active_tactic_ids[-1]
        self._initialize_progress_trackers()

    def provenance_state(self) -> ProvenanceState:
        """Return the underlying provenance graph (read-only usage expected)."""
        return self._provenance

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        del seed, options
        self._domain_engine.reset()
        self._provenance = ProvenanceState()
        self._step_count = 0
        self._initialize_progress_trackers()
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

        classification = action.tactic_category or "neutral"
        if self._dependencies:
            cls = self._dependencies.classify_tactic(stage_id, action.tactic)
            if cls:
                classification = cls
        if classification == "neutral":
            normalized = action.tactic.lower()
            if normalized in self._mandatory_targets.get(stage_id, set()):
                classification = "mandatory"
            elif normalized in self._optional_targets.get(stage_id, set()):
                classification = "optional"

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

        self._register_tactic_completion(stage_id, action.tactic, classification, reward_breakdown)

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
        technique_instances = self._instance_library.instances_for(action.technique_id)
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
            reward_breakdown["utility"] += 1.0
        if made_progress:
            self._mark_progress()

    def _register_tactic_completion(
        self,
        stage: int,
        tactic: str,
        classification: str,
        reward_breakdown: Dict[str, float],
    ) -> None:
        normalized = tactic.lower()
        if classification == "mandatory":
            targets = self._mandatory_targets.get(stage, set())
            if not targets:
                self._complete_stage(stage, reward_breakdown, add_pending=False)
                return
            progress = self._mandatory_progress.setdefault(stage, set())
            if normalized not in progress:
                progress.add(normalized)
                reward_breakdown["utility"] += self._mandatory_tactic_reward
                self._mark_progress()
            if targets.issubset(progress):
                self._complete_stage(stage, reward_breakdown, add_pending=True)
        elif classification == "optional":
            progress = self._optional_progress.setdefault(stage, set())
            if normalized not in progress:
                progress.add(normalized)
                reward_breakdown["utility"] += self._optional_tactic_reward
                self._mark_progress()

    def _complete_stage(
        self,
        stage: int,
        reward_breakdown: Optional[Dict[str, float]],
        add_pending: bool,
    ) -> None:
        if stage in self._completed_stages:
            return
        self._completed_stages.add(stage)
        if add_pending:
            if reward_breakdown is not None:
                reward_breakdown["structure"] += self._stage_completion_bonus
            self._pending_stage_completion[stage] = self._pending_stage_completion.get(stage, 0) + 1
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
        if stage in self._completed_stages:
            return
        targets = self._mandatory_targets.get(stage, set())
        if targets:
            return
        self._completed_stages.add(stage)
        self._advance_after_completion(stage, reward_breakdown)

    def _unlocked_stages(self) -> Tuple[int, ...]:
        return tuple(self._active_tactic_ids[idx] for idx in sorted(self._available_indices))

    def _current_focus_stage(self) -> int:
        for stage in self._active_tactic_ids:
            if stage not in self._completed_stages:
                return stage
        return self._active_tactic_ids[-1]

    def requires_mandatory_completion(self, stage: int) -> bool:
        return bool(self._mandatory_targets.get(stage, set()))

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
            mask = self._action_space_helper.mask_for_stage(mask_stages[0])
        else:
            mask = self._action_space_helper.mask_for_stages(mask_stages)
        return AttackObservation(
            topology=self._topology,
            provenance=self._provenance,
            stage=self._current_focus_stage(),
            step_count=self._step_count,
            action_mask=mask,
            reward_breakdown=dict(reward_breakdown),
        )
