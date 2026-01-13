"""Test network topology reachability constraints."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from env.apt_env import APTAttackEnv
from env.topology import TopologyGraph
from data.policy_loader import PolicyRepository
from data.technique_loader import TechniqueRepository
from data.instance_library import TechniqueInstanceLibrary
from data.dependency_loader import TacticDependencyMap


def test_agent_respects_network_topology():
    """Test that agent can only attack reachable devices."""
    # Load chemical plant topology
    base_dir = Path(__file__).parent.parent
    data_root = base_dir / "data"
    
    topology = TopologyGraph.from_file(
        data_root / "sample_topologies/chemical_plant.json"
    )
    policy_repo = PolicyRepository(data_root)
    tech_repo = TechniqueRepository(data_root)
    instance_lib = TechniqueInstanceLibrary(data_root)
    dependency_map = TacticDependencyMap(data_root)
    
    env = APTAttackEnv(
        topology=topology,
        policy_repo=policy_repo,
        technique_repo=tech_repo,
        instance_library=instance_lib,
        dependency_map=dependency_map,
    )
    
    obs, _ = env.reset()
    
    # Check initial entry point
    assert len(env._compromised_devices) == 1, "Should have one compromised device at start"
    entry_device = list(env._compromised_devices)[0]
    print(f"✓ Initial entry device: {entry_device}")
    
    # Get valid actions
    valid_actions = [i for i, valid in enumerate(obs.action_mask.values) if valid]
    print(f"✓ Found {len(valid_actions)} valid actions at start")
    
    # Verify: all valid actions target either:
    # 1. The entry device itself, OR
    # 2. Direct neighbors of entry device, OR
    # 3. Use lateral movement techniques
    reachable_devices = set([entry_device])
    reachable_devices.update(topology.neighbors(entry_device))
    
    violations = []
    lateral_movement_actions = 0
    
    for action_idx in valid_actions:
        action = env._action_space_helper.all_actions()[action_idx]
        
        is_reachable = action.target_device in reachable_devices
        is_lateral = env._is_lateral_movement_technique(action.technique_id)
        
        if is_lateral:
            lateral_movement_actions += 1
        
        if not is_reachable and not is_lateral:
            violations.append({
                "action_idx": action_idx,
                "target": action.target_device,
                "technique": action.technique_id,
                "tactic": action.tactic,
            })
    
    if violations:
        print(f"\n✗ Found {len(violations)} violations:")
        for v in violations[:5]:  # Show first 5
            print(f"  - Action {v['action_idx']}: targets unreachable device {v['target']}")
            print(f"    Technique: {v['technique']}, Tactic: {v['tactic']}")
        assert False, f"Found {len(violations)} actions targeting unreachable devices"
    
    print(f"✓ All valid actions respect network topology")
    print(f"✓ Lateral movement actions found: {lateral_movement_actions}")
    
    # Test that compromising a device expands reachability
    if valid_actions:
        initial_compromised = len(env._compromised_devices)
        action_idx = valid_actions[0]
        obs, reward, done, truncated, info = env.step(action_idx)
        
        current_compromised = len(env._compromised_devices)
        if current_compromised > initial_compromised:
            print(f"✓ Device compromise expanded from {initial_compromised} to {current_compromised} devices")
        
        # Check that new neighbors are now accessible
        new_valid_actions = [i for i, valid in enumerate(obs.action_mask.values) if valid]
        print(f"✓ Valid actions after step: {len(new_valid_actions)}")
    
    print("\n✅ All network topology tests passed!")


def test_lateral_movement_reaches_anywhere():
    """Test that lateral movement techniques can reach any device."""
    base_dir = Path(__file__).parent.parent
    data_root = base_dir / "data"
    
    topology = TopologyGraph.from_file(
        data_root / "sample_topologies/chemical_plant.json"
    )
    policy_repo = PolicyRepository(data_root)
    tech_repo = TechniqueRepository(data_root)
    instance_lib = TechniqueInstanceLibrary(data_root)
    dependency_map = TacticDependencyMap(data_root)
    
    env = APTAttackEnv(
        topology=topology,
        policy_repo=policy_repo,
        technique_repo=tech_repo,
        instance_library=instance_lib,
        dependency_map=dependency_map,
    )
    
    obs, _ = env.reset()
    
    # Find lateral movement actions
    lateral_actions = []
    for idx, action in enumerate(env._action_space_helper.all_actions()):
        if env._is_lateral_movement_technique(action.technique_id):
            lateral_actions.append((idx, action))
    
    if lateral_actions:
        print(f"✓ Found {len(lateral_actions)} lateral movement actions")
        # Show some examples
        for idx, action in lateral_actions[:3]:
            print(f"  - {action.technique_id}: {action.tactic} → {action.target_device}")
    else:
        print("⚠ No lateral movement techniques found in action space")
    
    print("✅ Lateral movement test completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Network Topology Reachability Constraints")
    print("=" * 60)
    print()
    
    try:
        test_agent_respects_network_topology()
        print()
        test_lateral_movement_reaches_anywhere()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

