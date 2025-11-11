"""Tests for convenience helpers exposed by the RL environment."""

from __future__ import annotations

from pathlib import Path
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


def _build_env() -> APTAttackEnv:
    policy_repo = PolicyRepository(DATA_ROOT)
    technique_repo = TechniqueRepository(DATA_ROOT)
    instance_lib = TechniqueInstanceLibrary(DATA_ROOT)
    topology = TopologyGraph.from_file(TOPOLOGY_PATH)
    dependency_map = TacticDependencyMap(DATA_ROOT)
    return APTAttackEnv(
        topology=topology,
        policy_repo=policy_repo,
        technique_repo=technique_repo,
        instance_library=instance_lib,
        dependency_map=dependency_map,
        stage_patience=5,
    )


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
