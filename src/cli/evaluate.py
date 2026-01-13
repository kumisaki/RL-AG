"""Evaluate trained PPO policies and export generated attack paths."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from env import APTAttackEnv, TopologyGraph, AttackMacroAction
from env.provenance import ProvenanceEdge, ProvenanceNode
from models import ActorCriticPolicy, EncoderConfig, PolicyConfig
from models.features import build_relation_vocab, encode_observation
from training.config import CurriculumConfig, OptimConfig, TrainingConfig


def _safe_checkpoint_load(
    checkpoint: Path,
    map_location: str,
    allow_unsafe: bool,
) -> Dict[str, Any]:
    if allow_unsafe:
        return torch.load(checkpoint, map_location=map_location)
    try:
        return torch.load(checkpoint, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Secure checkpoint loading requires torch>=2.0. "
            "Re-run with --allow-unsafe-checkpoint to fall back to pickle-based torch.load."
        ) from exc


def _load_model_weights(
    policy_model: ActorCriticPolicy,
    checkpoint: Path,
    device: str,
    allow_unsafe_checkpoint: bool,
) -> None:
    payload = _safe_checkpoint_load(checkpoint, map_location=device, allow_unsafe=allow_unsafe_checkpoint)
    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        # Allow checkpoints that store raw state_dict directly.
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a valid model state_dict.")
    policy_model.load_state_dict(state_dict)
    policy_model.eval()


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
    TopologyGraph,
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
    return env, policy_model, policies, relation_vocab, topology


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
    allow_unsafe_checkpoint: bool = False,
) -> Dict[str, float]:
    env, policy_model, policies, relation_vocab, _ = _load_components(
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
    _load_model_weights(
        policy_model,
        checkpoint=checkpoint,
        device=device,
        allow_unsafe_checkpoint=allow_unsafe_checkpoint,
    )

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
    allow_unsafe_checkpoint: bool = False,
    top_k: Optional[int] = None,
    require_plc_impact: bool = False,
) -> List[Path]:
    env, policy_model, policies, relation_vocab, topology_graph = _load_components(
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
    _load_model_weights(
        policy_model,
        checkpoint=checkpoint,
        device=device,
        allow_unsafe_checkpoint=allow_unsafe_checkpoint,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    episode_records = []
    for episode_idx in range(episodes):
        episode_result = _run_episode(
            env, policy_model, relation_vocab, device, deterministic=True
        )
        graph_payload = _provenance_to_json(env.provenance_state())
        impact_device = _detect_plc_impact(episode_result["actions"], topology_graph)
        episode_records.append(
            {
                "episode": episode_result,
                "graph": graph_payload,
                "index": episode_idx,
                "impact_device": impact_device,
            }
        )

    selected_records = episode_records
    if require_plc_impact:
        selected_records = [record for record in selected_records if record["impact_device"]]

    if top_k is not None:
        selected_records = sorted(
            selected_records,
            key=lambda rec: rec["episode"]["reward"],
            reverse=True,
        )[:top_k]

    if not selected_records:
        print(
            "No attack paths matched the selection criteria; nothing exported.")
        return generated_files

    for rank, record in enumerate(selected_records, start=1):
        episode_result = record["episode"]
        output_path = output_dir / f"episode_{record['index']}.json"
        payload = {
            "reward": episode_result["reward"],
            "length": episode_result["length"],
            "techniques": sorted(episode_result["techniques"]),
            "actions": [asdict(action) for action in episode_result["actions"]],
            "provenance": record["graph"],
            "plc_impact": bool(record["impact_device"]),
        }
        if record["impact_device"]:
            payload["impact_device"] = record["impact_device"]
        if top_k is not None:
            payload["rank"] = rank
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
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


def _detect_plc_impact(
    actions: Iterable[AttackMacroAction],
    topology: TopologyGraph,
) -> Optional[Dict[str, str]]:
    for action in actions:
        tactic_name = action.tactic.lower()
        if "impact" not in tactic_name:
            continue
        try:
            device = topology.get_node(action.target_device)
        except KeyError:
            continue
        if device.device_type.lower() != "plc":
            continue
        return {
            "device_id": device.device_id,
            "device_name": device.name,
            "device_type": device.device_type,
            "technique_id": action.technique_id,
            "tactic": action.tactic,
        }
    return None


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
        default=Path("data/sample_topologies/chemical_plant.json"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--stage-patience", type=int, default=16)
    parser.add_argument("--stage-completion-bonus", type=float, default=5.0)
    parser.add_argument("--stage-transition-bonus", type=float, default=1.0)
    parser.add_argument("--techniques-per-policy", type=int, default=3)
    parser.add_argument("--targets-per-technique", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only export the top-K attack paths by reward (after optional filtering).",
    )
    parser.add_argument(
        "--require-plc-impact",
        action="store_true",
        help="Only export attack paths that reach an impact action against a PLC device.",
    )
    parser.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="Permit legacy torch.load behaviour (executes pickle). "
        "Secure checkpoint loading is used by default.",
    )
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
        allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
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
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
            top_k=args.top_k,
            require_plc_impact=args.require_plc_impact,
        )
        print(f"Exported {len(files)} attack paths to {args.export_dir}")


if __name__ == "__main__":
    main()
