"""Dataclass configurations for PPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple


@dataclass
class OptimConfig:
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = 1.0


@dataclass
class CurriculumConfig:
    stages: Tuple[int, ...]
    phase_lengths: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.stages) != len(self.phase_lengths):
            raise ValueError("Curriculum stages and phase lengths must align")


@dataclass
class TrainingConfig:
    total_steps: int = 500_000
    rollout_length: int = 512
    minibatch_size: int = 256
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.3
    vf_coef: float = 0.5
    entropy_coef: float = 0.02
    target_kl: Optional[float] = 0.015
    device: str = "cpu"
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    curriculum: Optional[CurriculumConfig] = None
    seed: Optional[int] = None
    normalize_rewards: bool = False
