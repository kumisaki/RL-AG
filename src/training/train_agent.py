"""Command-line entry point for PPO training on the APT attack environment."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from data import (
    PolicyRepository,
    TechniqueRepository,
    TechniqueInstanceLibrary,
    EntityAlphabet,
    TacticDependencyMap,
)
from env import APTAttackEnv, TopologyGraph
from models import ActorCriticPolicy, EncoderConfig, PolicyConfig
from models.features import build_relation_vocab
from training.config import CurriculumConfig, OptimConfig, TrainingConfig
from training.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for IIoT attack path generation.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Path to data directory.")
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path("data/sample_topologies/chemical_plant.json"),
        help="Topology JSON file.",
    )
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--rollout-length", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--curriculum-stages",
        type=str,
        default="",
        help="Comma-separated stage ids for curriculum (e.g. '1,1,2,3,4').",
    )
    parser.add_argument(
        "--curriculum-lengths",
        type=str,
        default="",
        help="Comma-separated rollout counts for each curriculum phase (e.g. '5000,10000,15000').",
    )
    parser.add_argument(
        "--use-default-curriculum",
        action="store_true",
        help="Enable the built-in stage-by-stage curriculum when explicit stages are not provided.",
    )
    parser.add_argument("--stage-patience", type=int, default=16)
    parser.add_argument("--stage-completion-bonus", type=float, default=5.0)
    parser.add_argument("--stage-transition-bonus", type=float, default=1.0)
    parser.add_argument("--techniques-per-policy", type=int, default=3)
    parser.add_argument("--targets-per-technique", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--normalize-rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable running-stat reward normalization (default: enabled).",
    )
    return parser.parse_args()


def build_curriculum(args: argparse.Namespace) -> Optional[CurriculumConfig]:
    if args.curriculum_stages:
        stage_values = tuple(int(x.strip()) for x in args.curriculum_stages.split(",") if x.strip())
        length_values = tuple(int(x.strip()) for x in args.curriculum_lengths.split(",") if x.strip())
        if not length_values:
            raise ValueError("Curriculum lengths must be provided when curriculum stages are set.")
        return CurriculumConfig(stages=stage_values, phase_lengths=length_values)
    if args.use_default_curriculum:
        default_stages = tuple(range(1, 15))
        default_lengths = tuple(20_000 for _ in default_stages)
        return CurriculumConfig(stages=default_stages, phase_lengths=default_lengths)
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    curriculum = build_curriculum(args)
    set_seed(args.seed)

    data_root = args.data_root
    policy_repo = PolicyRepository(data_root)
    technique_repo = TechniqueRepository(data_root)
    instance_lib = TechniqueInstanceLibrary(data_root)
    topology = TopologyGraph.from_file(args.topology)
    dependency_map = TacticDependencyMap(data_root)

    env = APTAttackEnv(
        topology=topology,
        policy_repo=policy_repo,
        technique_repo=technique_repo,
        instance_library=instance_lib,
        dependency_map=dependency_map,
        stage_patience=args.stage_patience,
        stage_completion_bonus=args.stage_completion_bonus,
        stage_transition_bonus=args.stage_transition_bonus,
        techniques_per_policy=args.techniques_per_policy,
        targets_per_technique=args.targets_per_technique,
        max_steps=args.max_steps,
    )

    policies = list(policy_repo.iter_policies())
    relation_vocab = build_relation_vocab(policies)

    encoder_config = EncoderConfig(
        input_dim=len(EntityAlphabet),
        num_relations=len(relation_vocab),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    policy_config = PolicyConfig(
        encoder=encoder_config,
        action_dim=env.action_count(),
    )
    policy_model = ActorCriticPolicy(policy_config)

    training_config = TrainingConfig(
        total_steps=args.total_steps,
        rollout_length=args.rollout_length,
        minibatch_size=args.minibatch_size,
        ppo_epochs=args.ppo_epochs,
        device=args.device,
        entropy_coef=args.entropy_coef,
        curriculum=curriculum,
        seed=args.seed,
        normalize_rewards=args.normalize_rewards,
    )

    optim_config = OptimConfig(learning_rate=args.learning_rate)

    trainer = PPOTrainer(
        env=env,
        policy=policy_model,
        training_cfg=training_config,
        optim_cfg=optim_config,
        policies=policies,
    )
    trainer.train()


if __name__ == "__main__":
    main()
