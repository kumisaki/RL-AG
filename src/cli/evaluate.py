"""Evaluate trained PPO policies and export generated attack paths."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.distributions import Categorical

from data import (
    PolicyRepository,
    TechniqueRepository,
    TechniqueInstanceLibrary,
    EntityAlphabet,
    TacticDependencyMap,
)
from data.models import PolicyDefinition
from env import APTAttackEnv, TopologyGraph
from env.provenance import ProvenanceEdge, ProvenanceNode
from models import ActorCriticPolicy, EncoderConfig, PolicyConfig
from models.features import build_relation_vocab, encode_observation
from training.config import CurriculumConfig, OptimConfig, TrainingConfig


def _load_components(
    data_root: Path,
    topology_path: Path,
    device: str,
    hidden_dim: int,
    num_layers: int,
    stage_patience: int,
    stage_completion_bonus: float,
    stage_transition_bonus: float,
    techniques_per_policy: Optional[int],
    targets_per_technique: Optional[int],
    max_steps: int,
) -> Tuple[
    APTAttackEnv,
    ActorCriticPolicy,
    List[PolicyDefinition],
    Dict[str, int],
]:
    policy_repo = PolicyRepository(data_root)
    technique_repo = TechniqueRepository(data_root)
    instance_lib = TechniqueInstanceLibrary(data_root)
    topology = TopologyGraph.from_file(topology_path)
    dependency_map = TacticDependencyMap(data_root)
    env = APTAttackEnv(
        topology=topology,
        policy_repo=policy_repo,
        technique_repo=technique_repo,
        instance_library=instance_lib,
        dependency_map=dependency_map,
        stage_patience=stage_patience,
        stage_completion_bonus=stage_completion_bonus,
        stage_transition_bonus=stage_transition_bonus,
        techniques_per_policy=techniques_per_policy,
        targets_per_technique=targets_per_technique,
        max_steps=max_steps,
    )
    policies = list(policy_repo.iter_policies())
    relation_vocab = build_relation_vocab(policies)
    encoder_config = EncoderConfig(
        input_dim=len(EntityAlphabet),
        num_relations=len(relation_vocab),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    policy_config = PolicyConfig(encoder=encoder_config, action_dim=env.action_count())
    policy_model = ActorCriticPolicy(policy_config).to(device)
    return env, policy_model, policies, relation_vocab


def evaluate_policy(
    checkpoint: Path,
    data_root: Path,
    topology: Path,
    episodes: int,
    device: str = "cpu",
    hidden_dim: int = 256,
    num_layers: int = 3,
    deterministic: bool = True,
    stage_patience: int = 8,
    stage_completion_bonus: float = 5.0,
    stage_transition_bonus: float = 1.0,
    techniques_per_policy: Optional[int] = 3,
    targets_per_technique: Optional[int] = 5,
    max_steps: int = 200,
) -> Dict[str, float]:
    env, policy_model, policies, relation_vocab = _load_components(
        data_root,
        topology,
        device,
        hidden_dim,
        num_layers,
        stage_patience=stage_patience,
        stage_completion_bonus=stage_completion_bonus,
        stage_transition_bonus=stage_transition_bonus,
        techniques_per_policy=techniques_per_policy,
        targets_per_technique=targets_per_technique,
        max_steps=max_steps,
    )
    state_dict = torch.load(checkpoint, map_location=device)
    policy_model.load_state_dict(state_dict["model_state_dict"])
    policy_model.eval()

    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "terminal_stage_hits": 0,
        "technique_diversity": set(),
    }

    for _ in range(episodes):
        episode_result = _run_episode(
            env, policy_model, relation_vocab, device, deterministic=deterministic
        )
        metrics["episode_rewards"].append(episode_result["reward"])
        metrics["episode_lengths"].append(episode_result["length"])
        metrics["technique_diversity"].update(episode_result["techniques"])
        if episode_result["terminated"]:
            metrics["terminal_stage_hits"] += 1

    summary = {
        "avg_reward": float(sum(metrics["episode_rewards"]) / len(metrics["episode_rewards"])),
        "avg_length": float(sum(metrics["episode_lengths"]) / len(metrics["episode_lengths"])),
        "success_rate": metrics["terminal_stage_hits"] / episodes,
        "technique_diversity": len(metrics["technique_diversity"]),
    }
    return summary


def generate_attack_paths(
    checkpoint: Path,
    data_root: Path,
    topology: Path,
    output_dir: Path,
    episodes: int = 10,
    device: str = "cpu",
    hidden_dim: int = 256,
    num_layers: int = 3,
    stage_patience: int = 8,
    stage_completion_bonus: float = 5.0,
    stage_transition_bonus: float = 1.0,
    techniques_per_policy: Optional[int] = 3,
    targets_per_technique: Optional[int] = 5,
    max_steps: int = 200,
) -> List[Path]:
    env, policy_model, policies, relation_vocab = _load_components(
        data_root,
        topology,
        device,
        hidden_dim,
        num_layers,
        stage_patience=stage_patience,
        stage_completion_bonus=stage_completion_bonus,
        stage_transition_bonus=stage_transition_bonus,
        techniques_per_policy=techniques_per_policy,
        targets_per_technique=targets_per_technique,
        max_steps=max_steps,
    )
    state_dict = torch.load(checkpoint, map_location=device)
    policy_model.load_state_dict(state_dict["model_state_dict"])
    policy_model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    for episode_idx in range(episodes):
        episode_result = _run_episode(
            env, policy_model, relation_vocab, device, deterministic=True
        )
        graph_payload = _provenance_to_json(env.provenance_state())
        output_path = output_dir / f"episode_{episode_idx}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "reward": episode_result["reward"],
                    "length": episode_result["length"],
                    "actions": [asdict(action) for action in episode_result["actions"]],
                    "provenance": graph_payload,
                },
                handle,
                indent=2,
            )
        generated_files.append(output_path)
    return generated_files


def _run_episode(
    env: APTAttackEnv,
    policy: ActorCriticPolicy,
    relation_vocab: Dict[str, int],
    device: str,
    deterministic: bool,
) -> Dict[str, object]:
    observation, _ = env.reset()
    total_reward = 0.0
    steps = 0
    techniques = set()
    actions = []
    done = False
    while not done:
        batch = encode_observation(observation, relation_vocab).to(torch.device(device))
        with torch.no_grad():
            output = policy.forward(batch)
            mask_tensor = torch.tensor(
                observation.action_mask.values,
                dtype=torch.bool,
                device=output.logits.device,
            )
            logits = output.logits.clone()
            if torch.any(mask_tensor):
                logits[~mask_tensor] = float("-inf")
            else:
                mask_tensor = torch.ones_like(mask_tensor, dtype=torch.bool)
            if deterministic:
                action_idx = int(torch.argmax(logits).item())
            else:
                distribution = Categorical(logits=logits)
                action_idx = int(distribution.sample().item())
        observation, reward, terminated, truncated, info = env.step(action_idx)
        total_reward += reward
        steps += 1
        action = info.get("action")
        if action:
            techniques.add(action.technique_id)
            actions.append(action)
        done = terminated or truncated
    return {
        "reward": total_reward,
        "length": steps,
        "techniques": techniques,
        "actions": actions,
        "terminated": terminated,
    }


def _provenance_to_json(state) -> Dict[str, object]:
    nodes = []
    for node in state.iter_nodes():
        nodes.append(
            {
                "id": node.node_id,
                "entity_type": node.entity_type,
                "host_device": node.host_device,
                "name": node.name,
                "attributes": node.attributes,
            }
        )
    edges = []
    for edge in state.iter_edges():
        edges.append(
            {
                "id": edge.edge_id,
                "subject": edge.subject,
                "relation": edge.relation,
                "object": edge.obj,
                "technique_id": edge.technique_id,
                "policy": str(edge.policy_key) if edge.policy_key else None,
                "metadata": edge.metadata,
            }
        )
    return {"nodes": nodes, "edges": edges}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PPO policy and export attack paths.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path("data/sample_topologies/chemical_park_central.json"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--stage-patience", type=int, default=8)
    parser.add_argument("--stage-completion-bonus", type=float, default=5.0)
    parser.add_argument("--stage-transition-bonus", type=float, default=1.0)
    parser.add_argument("--techniques-per-policy", type=int, default=3)
    parser.add_argument("--targets-per-technique", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    summary = evaluate_policy(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        topology=args.topology,
        episodes=args.episodes,
        device=args.device,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        stage_patience=args.stage_patience,
        stage_completion_bonus=args.stage_completion_bonus,
        stage_transition_bonus=args.stage_transition_bonus,
        techniques_per_policy=args.techniques_per_policy,
        targets_per_technique=args.targets_per_technique,
        max_steps=args.max_steps,
    )
    print(json.dumps(summary, indent=2))
    if args.export_dir:
        files = generate_attack_paths(
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            topology=args.topology,
            output_dir=args.export_dir,
            episodes=args.episodes,
            device=args.device,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            stage_patience=args.stage_patience,
            stage_completion_bonus=args.stage_completion_bonus,
            stage_transition_bonus=args.stage_transition_bonus,
            techniques_per_policy=args.techniques_per_policy,
            targets_per_technique=args.targets_per_technique,
            max_steps=args.max_steps,
        )
        print(f"Exported {len(files)} attack paths to {args.export_dir}")


if __name__ == "__main__":
    main()
