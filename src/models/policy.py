"""Actor-critic policy head operating on relational graph embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from .encoders import GraphBatch, RelationalGraphEncoder, EncoderConfig


@dataclass
class PolicyConfig:
    encoder: EncoderConfig
    action_dim: int


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor


class ActorCriticPolicy(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.encoder = RelationalGraphEncoder(config.encoder)
        self.actor = nn.Linear(config.encoder.hidden_dim, config.action_dim)
        self.critic = nn.Linear(config.encoder.hidden_dim, 1)

    def forward(self, batch: GraphBatch) -> PolicyOutput:
        embedding = self.encoder(batch)
        logits = self.actor(embedding)
        value = self.critic(embedding).squeeze(-1)
        return PolicyOutput(logits=logits, value=value)

    def _masked_logits(self, logits: torch.Tensor, action_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if action_mask is None:
            return logits
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()
        if torch.any(action_mask):
            masked = logits.clone()
            masked[~action_mask] = float("-inf")
            return masked
        return logits

    def act(
        self,
        batch: GraphBatch,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward(batch)
        logits = self._masked_logits(output.logits, action_mask)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob, output.value
