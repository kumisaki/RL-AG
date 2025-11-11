"""Generate provenance graphs using heuristic macro-action selection.

This utility is meant for quick inspection without needing a trained PPO checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import shutil
import subprocess

from data import (
    PolicyRepository,
    TechniqueRepository,
    TechniqueInstanceLibrary,
    TacticDependencyMap,
)
from env import APTAttackEnv, TopologyGraph
from env.actions import AttackMacroAction

from .evaluate import _provenance_to_json


def _build_env(
    data_root: Path,
    topology: Path,
    stage_patience: int,
    stage_completion_bonus: float,
    stage_transition_bonus: float,
    techniques_per_policy: Optional[int],
    targets_per_technique: Optional[int],
) -> APTAttackEnv:
    policy_repo = PolicyRepository(data_root)
    technique_repo = TechniqueRepository(data_root)
    instance_lib = TechniqueInstanceLibrary(data_root)
    topo = TopologyGraph.from_file(topology)
    dependency_map = TacticDependencyMap(data_root)
    return APTAttackEnv(
        topology=topo,
        policy_repo=policy_repo,
        technique_repo=technique_repo,
        instance_library=instance_lib,
        dependency_map=dependency_map,
        stage_patience=stage_patience,
        stage_completion_bonus=stage_completion_bonus,
        stage_transition_bonus=stage_transition_bonus,
        techniques_per_policy=techniques_per_policy,
        targets_per_technique=targets_per_technique,
    )


def _select_action(actions: Tuple[AttackMacroAction, ...], strategy: str) -> AttackMacroAction:
    if not actions:
        raise RuntimeError("No available macro-actions for the current stage")
    if strategy == "random":
        return random.choice(actions)
    return actions[0]


def _run_episode(env: APTAttackEnv, strategy: str, max_steps: int) -> Dict[str, object]:
    observation, _ = env.reset()
    total_reward = 0.0
    steps = 0
    actions_taken: List[AttackMacroAction] = []
    done = False
    while not done and steps < max_steps:
        available = env.available_actions()
        chosen = _select_action(available, strategy)
        observation, reward, terminated, truncated, info = env.step(chosen.index)
        total_reward += reward
        steps += 1
        action = info.get("action")
        if action:
            actions_taken.append(action)
        done = terminated or truncated
    return {
        "reward": total_reward,
        "length": steps,
        "actions": [asdict(action) for action in actions_taken],
    }


def _write_graphviz(graph_payload: Dict[str, object], dot_path: Path) -> List[Path]:
    nodes: List[Dict[str, object]] = graph_payload["nodes"]  # type: ignore[assignment]
    edges: List[Dict[str, object]] = graph_payload["edges"]  # type: ignore[assignment]
    lines = ["digraph provenance {", '  rankdir=LR;']
    for node in nodes:
        label = f"{node['entity_type']}\\n{node.get('name') or node['id']}"
        lines.append(f'  "{node["id"]}" [label="{label}"];')
    for edge in edges:
        meta = edge.get("technique_id") or ""
        lines.append(
            f'  "{edge["subject"]}" -> "{edge["object"]}" '
            f'[label="{edge["relation"]}\\n{meta}"];'
        )
    lines.append("}")
    dot_path.write_text("\n".join(lines), encoding="utf-8")
    artifacts = [dot_path]
    dot_bin = shutil.which("dot")
    if dot_bin:
        svg_path = dot_path.with_suffix(".svg")
        subprocess.run(
            [dot_bin, "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=False,
        )
        artifacts.append(svg_path)
    return artifacts


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate sample provenance graphs without PPO checkpoints.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Path to TAGAPT-derived data directory.")
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path("data/sample_topologies/chemical_park_central.json"),
        help="Topology JSON describing the target IIoT environment.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to generate.")
    parser.add_argument("--strategy", choices=("first", "random"), default="first", help="Macro-action selection strategy.")
    parser.add_argument("--max-steps", type=int, default=50, help="Safety cap on episode length.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility when using random strategy.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_runs"),
        help="Directory where provenance graphs will be stored.",
    )
    parser.add_argument(
        "--graphviz",
        action="store_true",
        help="Emit Graphviz DOT/SVG artifacts alongside the JSON output.",
    )
    parser.add_argument("--stage-patience", type=int, default=8)
    parser.add_argument("--stage-completion-bonus", type=float, default=5.0)
    parser.add_argument("--stage-transition-bonus", type=float, default=1.0)
    parser.add_argument("--techniques-per-policy", type=int, default=3)
    parser.add_argument("--targets-per-technique", type=int, default=5)
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    random.seed(args.seed)
    env = _build_env(
        args.data_root,
        args.topology,
        stage_patience=args.stage_patience,
        stage_completion_bonus=args.stage_completion_bonus,
        stage_transition_bonus=args.stage_transition_bonus,
        techniques_per_policy=args.techniques_per_policy,
        targets_per_technique=args.targets_per_technique,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    for episode_idx in range(args.episodes):
        summary = _run_episode(env, strategy=args.strategy, max_steps=args.max_steps)
        graph_payload = _provenance_to_json(env.provenance_state())
        output_path = args.output_dir / f"sample_episode_{episode_idx}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "reward": summary["reward"],
                    "length": summary["length"],
                    "actions": summary["actions"],
                    "provenance": graph_payload,
                },
                handle,
                indent=2,
            )
        generated.append(output_path)
        if args.graphviz:
            dot_path = output_path.with_suffix(".dot")
            artifacts = _write_graphviz(graph_payload, dot_path)
            generated.extend(artifacts)
    print(f"Generated {len(generated)} provenance artifacts under {args.output_dir}")


if __name__ == "__main__":
    main()
