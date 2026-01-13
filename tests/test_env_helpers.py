"""Tests for convenience helpers exposed by the RL environment."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import (  # type: ignore  # noqa: E402
    PolicyRepository,
    TechniqueRepository,
    TechniqueInstanceLibrary,
    TacticDependencyMap,
)
from env import APTAttackEnv, TopologyGraph  # type: ignore  # noqa: E402


DATA_ROOT = ROOT / "data"
TOPOLOGY_PATH = DATA_ROOT / "sample_topologies" / "chemical_park_central.json"
CHEMICAL_PLANT_PATH = DATA_ROOT / "sample_topologies" / "chemical_plant.json"


def _build_env(topology_path: Path = TOPOLOGY_PATH) -> APTAttackEnv:
    policy_repo = PolicyRepository(DATA_ROOT)
    technique_repo = TechniqueRepository(DATA_ROOT)
    instance_lib = TechniqueInstanceLibrary(DATA_ROOT)
    topology = TopologyGraph.from_file(topology_path)
    dependency_map = TacticDependencyMap(DATA_ROOT)
    return APTAttackEnv(
        topology=topology,
        policy_repo=policy_repo,
        technique_repo=technique_repo,
        instance_library=instance_lib,
        dependency_map=dependency_map,
        stage_patience=5,
    )


class RecordingInstanceLibrary(TechniqueInstanceLibrary):
    def __init__(self, data_root: Path) -> None:
        super().__init__(data_root)
        self.calls: list[tuple[str, Optional[str]]] = []

    def instances_for(self, technique_id: str, platform: Optional[str] = None):
        self.calls.append((technique_id, platform))
        return super().instances_for(technique_id, platform)


def test_action_count_matches_action_space() -> None:
    env = _build_env()
    count = env.action_count()
    assert count > 0
    if env.action_space is not None:
        assert env.action_space.n == count


def test_stage_order_override_limits_available_actions() -> None:
    env = _build_env()
    policies = list(PolicyRepository(DATA_ROOT).iter_policies())
    stages = []
    for policy in policies:
        if policy.key.stage not in stages:
            stages.append(policy.key.stage)
    env.set_stage_order((stages[0],))
    actions = env.available_actions()
    assert actions
    assert all(action.stage == stages[0] for action in actions)
    # Expanding back to the default order should not raise
    env.set_stage_order(tuple(stages))


def test_stage_unlocks_after_mandatory_tactics() -> None:
    env = _build_env()
    env.reset()
    stage1_actions = env.available_actions()
    assert all(action.stage == 1 for action in stage1_actions)
    action = next(a for a in env.available_actions() if a.tactic == "Reconnaissance")
    env.step(action.index)
    unlocked_actions = env.available_actions()
    assert any(action.stage == 2 for action in unlocked_actions)


def test_provenance_state_updates_after_step() -> None:
    env = _build_env()
    env.reset()
    assert not tuple(env.provenance_state().iter_nodes())
    first_action = env.available_actions()[0]
    env.step(first_action.index)
    assert tuple(env.provenance_state().iter_nodes()), "expected provenance graph to grow after a step"


def test_optional_only_stage_allows_actions_and_explicit_skip() -> None:
    env = _build_env()
    env.reset()
    # Stage 4 in the sample dependency map has no mandatory tactics.
    optional_stage = 4
    for stage in range(1, optional_stage):
        env._complete_stage(stage, reward_breakdown=None, add_pending=False)  # type: ignore[attr-defined]
    actions = [a for a in env.available_actions() if a.stage == optional_stage]
    assert actions, "expected optional-stage actions to be available"
    skip_actions = [a for a in actions if getattr(a, "kind", "policy") == "skip"]
    assert skip_actions, "missing skip action for optional stage"
    optional_actions = [a for a in actions if getattr(a, "kind", "policy") == "policy"]
    assert optional_actions, "expected optional macro-actions to remain available"
    env.step(optional_actions[0].index)
    assert optional_stage in env._completed_stages  # type: ignore[attr-defined]
    # Skipping after completion should be a no-op but must not error.
    env.step(skip_actions[0].index)


def test_instances_use_device_platform_when_available() -> None:
    policy_repo = PolicyRepository(DATA_ROOT)
    technique_repo = TechniqueRepository(DATA_ROOT)
    instance_lib = RecordingInstanceLibrary(DATA_ROOT)
    topology = TopologyGraph.from_file(TOPOLOGY_PATH)
    for node_id in topology.node_ids():
        node = topology.get_node(node_id)
        node.platform = "windows"
    dependency_map = TacticDependencyMap(DATA_ROOT)
    env = APTAttackEnv(
        topology=topology,
        policy_repo=policy_repo,
        technique_repo=technique_repo,
        instance_library=instance_lib,
        dependency_map=dependency_map,
    )
    env.reset()
    action = env.available_actions()[0]
    env.step(action.index)
    assert instance_lib.calls, "expected TechniqueInstanceLibrary to be queried"
    assert instance_lib.calls[-1][1] == "windows"


def test_completed_stage_stops_exposing_actions() -> None:
    env = _build_env()
    env.reset()
    first_action = env.available_actions()[0]
    env.step(first_action.index)
    remaining_stages = {action.stage for action in env.available_actions()}
    assert 1 not in remaining_stages, "completed stages should no longer provide actions"


def test_late_stage_multiplier_exceeds_early_stage() -> None:
    env = _build_env(CHEMICAL_PLANT_PATH)
    env.reset()
    early = env._stage_reward_multiplier(1)  # type: ignore[attr-defined]
    late = env._stage_reward_multiplier(14)  # type: ignore[attr-defined]
    assert late > early


def test_impact_stage_awards_large_bonus() -> None:
    env = _build_env(CHEMICAL_PLANT_PATH)
    env.set_stage_order((14,))
    env.reset()
    actions = env.available_actions()
    assert actions, "expected stage 14 actions to be available"
    action = actions[0]
    assert action.stage == 14
    _, _, _, _, info = env.step(action.index)
    bonus = info["reward_breakdown"]["utility"]
    assert bonus >= env._impact_bonus - 1e-6  # type: ignore[attr-defined]
