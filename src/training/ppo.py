"""PPO training loop with curriculum support and reward logging."""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from data.models import PolicyDefinition
from env.apt_env import APTAttackEnv
from models import ActorCriticPolicy, GraphBatch
from models.features import build_relation_vocab, encode_observation
from training.config import CurriculumConfig, OptimConfig, TrainingConfig

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


@dataclass
class Transition:
    graph_batch: GraphBatch
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool
    terminated: bool
    action_mask: torch.Tensor


class PPOTrainer:
    """Implements on-policy PPO with optional curriculum staging."""

    def __init__(
        self,
        env: APTAttackEnv,
        policy: ActorCriticPolicy,
        training_cfg: TrainingConfig,
        optim_cfg: OptimConfig,
        policies: Iterable[PolicyDefinition],
    ) -> None:
        self.env = env
        self.policy = policy
        self.training_cfg = training_cfg
        self.optim_cfg = optim_cfg
        self.device = torch.device(training_cfg.device)
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=optim_cfg.learning_rate,
            weight_decay=optim_cfg.weight_decay,
        )
        self.scheduler = None
        self.relation_vocab = build_relation_vocab(policies)
        self.log_dir = training_cfg.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = training_cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir) if SummaryWriter is not None else None

        self.curriculum = training_cfg.curriculum
        self._curriculum_pointer = 0
        self._stage_rollouts = 0
        self._active_stage_subset: Tuple[int, ...] = tuple()
        self._stage_success_history: Dict[int, deque[float]] = {}
        self._success_window = 5
        self._success_threshold = 0.7
        self._curriculum_needs_reset = False
        self.reward_stats = RunningStats() if training_cfg.normalize_rewards else None
        if self.curriculum:
            self._apply_curriculum(force=True)

    def train(self) -> None:
        observation, _ = self.env.reset()
        self._curriculum_needs_reset = False
        next_value = torch.zeros(1)
        for step in range(0, self.training_cfg.total_steps, self.training_cfg.rollout_length):
            if self.curriculum:
                self._apply_curriculum()
                if self._curriculum_needs_reset:
                    observation, _ = self.env.reset()
                    self._curriculum_needs_reset = False
            transitions: List[Transition] = []
            for rollout_step in range(self.training_cfg.rollout_length):
                graph_batch = encode_observation(observation, self.relation_vocab)
                graph_batch = graph_batch.to(self.device)
                mask_tensor = torch.tensor(
                    observation.action_mask.values,
                    device=self.device,
                    dtype=torch.bool,
                )
                action, log_prob, value = self.policy.act(graph_batch, action_mask=mask_tensor)
                next_observation, reward, terminated, truncated, info = self.env.step(action.item())
                transitions.append(
                    Transition(
                        graph_batch=graph_batch,
                        action=action.detach().long(),
                        log_prob=log_prob.detach(),
                        value=value.detach().view(1),
                        reward=float(reward),
                        done=terminated or truncated,
                        terminated=bool(terminated),
                        action_mask=mask_tensor.detach().cpu(),
                    )
                )
                if terminated or truncated:
                    next_observation, _ = self.env.reset()
                observation = next_observation
            with torch.no_grad():
                next_batch = encode_observation(observation, self.relation_vocab).to(self.device)
                next_value = self.policy.forward(next_batch).value.detach().view(1)
            returns, advantages = self._compute_advantages(transitions, next_value)
            loss_metrics = self._ppo_update(transitions, returns, advantages)
            self._log_rollout(step, transitions, loss_metrics)
            if (step // self.training_cfg.rollout_length) % 10 == 0:
                self._save_checkpoint(step)
        if self.writer:
            self.writer.flush()
            self.writer.close()

    def _compute_advantages(
        self,
        transitions: Sequence[Transition],
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.tensor([t.reward for t in transitions], device=self.device)
        if self.reward_stats is not None:
            self.reward_stats.update(rewards)
            rewards = self.reward_stats.normalize(rewards)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=self.device)
        values = torch.cat([t.value for t in transitions] + [next_value], dim=0)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(transitions))):
            delta = (
                rewards[step]
                + self.training_cfg.gamma * values[step + 1] * (1.0 - dones[step])
                - values[step]
            )
            gae = (
                delta
                + self.training_cfg.gamma
                * self.training_cfg.gae_lambda
                * (1.0 - dones[step])
                * gae
            )
            advantages[step] = gae
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns.detach(), advantages.detach()

    def _ppo_update(
        self,
        transitions: Sequence[Transition],
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        indices = list(range(len(transitions)))
        policy_loss_values: List[float] = []
        value_loss_values: List[float] = []
        entropy_values: List[float] = []
        for epoch in range(self.training_cfg.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), self.training_cfg.minibatch_size):
                batch_indices = indices[start : start + self.training_cfg.minibatch_size]
                if not batch_indices:
                    continue
                policy_losses = []
                value_losses = []
                entropy_losses = []
                for idx in batch_indices:
                    transition = transitions[idx]
                    batch = transition.graph_batch
                    action = transition.action.to(self.device)
                    old_log_prob = transition.log_prob.to(self.device)
                    output = self.policy.forward(batch)
                    mask = transition.action_mask.to(self.device)
                    logits = output.logits.clone()
                    if torch.any(mask):
                        logits[~mask] = float("-inf")
                    else:
                        mask = torch.ones_like(mask, dtype=torch.bool)
                    distribution = torch.distributions.Categorical(logits=logits)
                    new_log_prob = distribution.log_prob(action)
                    entropy_losses.append(distribution.entropy())
                    ratio = torch.exp(new_log_prob - old_log_prob)
                    advantage = advantages[idx]
                    surrogate1 = ratio * advantage
                    surrogate2 = torch.clamp(
                        ratio,
                        1.0 - self.training_cfg.clip_coef,
                        1.0 + self.training_cfg.clip_coef,
                    ) * advantage
                    policy_losses.append(-torch.min(surrogate1, surrogate2))
                    value_loss = (output.value - returns[idx]).pow(2)
                    value_losses.append(value_loss)
                policy_mean = torch.stack(policy_losses).mean()
                value_mean = torch.stack(value_losses).mean()
                entropy_mean = torch.stack(entropy_losses).mean()
                loss = (
                    policy_mean
                    + self.training_cfg.vf_coef * value_mean
                    - self.training_cfg.entropy_coef * entropy_mean
                )
                self.optimizer.zero_grad()
                loss.backward()
                if self.optim_cfg.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.optim_cfg.clip_grad_norm)
                self.optimizer.step()
                policy_loss_values.append(float(policy_mean.detach()))
                value_loss_values.append(float(value_mean.detach()))
                entropy_values.append(float(entropy_mean.detach()))
        return {
            "policy_loss": sum(policy_loss_values) / len(policy_loss_values)
            if policy_loss_values
            else 0.0,
            "value_loss": sum(value_loss_values) / len(value_loss_values)
            if value_loss_values
            else 0.0,
            "entropy": sum(entropy_values) / len(entropy_values)
            if entropy_values
            else 0.0,
        }

    def _save_checkpoint(self, step: int) -> None:
        checkpoint_path = self.checkpoint_dir / f"ppo_step_{step}.pt"
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": step,
            },
            checkpoint_path,
        )

    def _apply_curriculum(self, force: bool = False) -> bool:
        assert self.curriculum is not None
        if self._curriculum_pointer >= len(self.curriculum.stages):
            self._curriculum_pointer = len(self.curriculum.stages) - 1
            return False
        stage_subset = self.curriculum.stages[: self._curriculum_pointer + 1]
        changed = force or stage_subset != self._active_stage_subset
        if changed:
            self.env.set_stage_order(stage_subset)
            self._active_stage_subset = stage_subset
            if force:
                self._stage_rollouts = 0
            self._curriculum_needs_reset = True
        return changed

    def _record_stage_success(
        self,
        stage_subset: Tuple[int, ...],
        stage_completions: Dict[int, int],
        done_rate: float,
    ) -> None:
        for stage in stage_subset:
            history = self._stage_success_history.get(stage)
            if history is None or history.maxlen != self._success_window:
                history = deque(maxlen=self._success_window)
                self._stage_success_history[stage] = history
            completion_hits = stage_completions.get(stage, 0)
            requires_completion = True
            if hasattr(self.env, "requires_mandatory_completion"):
                checker = getattr(self.env, "requires_mandatory_completion")
                if callable(checker):
                    requires_completion = bool(checker(stage))
            success_value = 1.0 if (completion_hits > 0 or not requires_completion) else 0.0
            if stage == stage_subset[-1]:
                success_value = max(success_value, min(done_rate, 1.0))
            history.append(success_value)

    def _maybe_advance_curriculum(
        self,
        stage_subset: Tuple[int, ...],
        global_step: int,
    ) -> None:
        assert self.curriculum is not None
        if self._curriculum_pointer >= len(self.curriculum.stages) - 1:
            return
        self._stage_rollouts += 1
        phase_rollouts = self.curriculum.phase_lengths[self._curriculum_pointer]
        min_rollouts = max(1, phase_rollouts)
        if self._stage_rollouts < min_rollouts:
            return
        current_stage = stage_subset[-1]
        history = self._stage_success_history.get(current_stage)
        if not history or len(history) < min(self._success_window, min_rollouts):
            return
        avg_success = sum(history) / len(history)
        if avg_success < self._success_threshold:
            return
        print(
            f"[Curriculum] Advancing after rollout {global_step}: "
            f"stage {current_stage} success={avg_success:.2f}"
        )
        self._curriculum_pointer = min(
            self._curriculum_pointer + 1, len(self.curriculum.stages) - 1
        )
        self._stage_rollouts = 0
        self._apply_curriculum(force=True)

    def _log_rollout(
        self,
        step: int,
        transitions: Sequence[Transition],
        loss_metrics: Dict[str, float],
    ) -> None:
        global_step = step // self.training_cfg.rollout_length
        rewards = [t.reward for t in transitions]
        avg_reward = sum(rewards) / len(rewards)
        done_rate = sum(1 for t in transitions if t.done) / len(transitions)
        impact_success = sum(1 for t in transitions if t.terminated) / len(transitions)
        stage_counts = {}
        stage_completions: Dict[int, int] = {}
        if hasattr(self.env, "consume_stage_visitation"):
            stage_counts = getattr(self.env, "consume_stage_visitation")()
        if hasattr(self.env, "consume_stage_completions"):
            stage_completions = getattr(self.env, "consume_stage_completions")()

        stage_subset: Optional[Tuple[int, ...]] = None
        if self.curriculum:
            stage_subset = self.curriculum.stages[: self._curriculum_pointer + 1]
            self._record_stage_success(stage_subset, stage_completions, done_rate)
            self._maybe_advance_curriculum(stage_subset, global_step)

        print(
            f"[Rollout {global_step}] avg_reward={avg_reward:.3f} done_rate={done_rate:.2f} "
            f"policy_loss={loss_metrics['policy_loss']:.3f} value_loss={loss_metrics['value_loss']:.3f} "
            f"entropy={loss_metrics['entropy']:.3f} impact_success={impact_success:.2f} "
            f"stages={stage_counts} completions={stage_completions}"
        )
        if not self.writer:
            return
        self.writer.add_scalar("reward/avg", avg_reward, global_step)
        self.writer.add_scalar("reward/done_rate", done_rate, global_step)
        self.writer.add_scalar("reward/impact_success_rate", impact_success, global_step)
        self.writer.add_scalar("loss/policy", loss_metrics["policy_loss"], global_step)
        self.writer.add_scalar("loss/value", loss_metrics["value_loss"], global_step)
        self.writer.add_scalar("policy/entropy", loss_metrics["entropy"], global_step)
        if stage_subset:
            current_stage = stage_subset[-1]
            history = self._stage_success_history.get(current_stage)
            if history:
                avg_success = sum(history) / len(history)
                self.writer.add_scalar("curriculum/current_stage_success", avg_success, global_step)
            self.writer.add_scalar("curriculum/pointer", self._curriculum_pointer, global_step)
class RunningStats:
    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, values: torch.Tensor) -> None:
        batch_mean = float(values.mean().item())
        batch_var = float(values.var(unbiased=False).item())
        batch_count = values.numel()
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count if total_count > 0 else 1.0
        self.count = total_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        std = (self.var + 1e-8) ** 0.5
        return (values - self.mean) / std
