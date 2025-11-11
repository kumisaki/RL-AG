"""Relational graph encoder for heterogeneous provenance graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class GraphBatch:
    """Container for batched graph inputs."""

    node_features: torch.Tensor  # [N, F]
    edge_index: torch.Tensor  # [2, E]
    edge_types: torch.Tensor  # [E]
    num_relations: int

    def to(self, device: torch.device) -> "GraphBatch":
        return GraphBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_types=self.edge_types.to(device),
            num_relations=self.num_relations,
        )


@dataclass
class EncoderConfig:
    input_dim: int
    num_relations: int
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.1


class RelGraphConvLayer(nn.Module):
    """Minimal relational graph convolution layer."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, bias: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.weight = nn.Parameter(torch.empty(num_relations, out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, batch: GraphBatch, node_repr: torch.Tensor) -> torch.Tensor:
        src, dst = batch.edge_index
        messages = torch.zeros(
            node_repr.size(0),
            self.out_dim,
            device=node_repr.device,
            dtype=node_repr.dtype,
        )
        for relation in range(batch.num_relations):
            mask = batch.edge_types == relation
            if not torch.any(mask):
                continue
            weight = self.weight[relation]
            rel_src = src[mask]
            rel_dst = dst[mask]
            transformed = node_repr[rel_src] @ weight.t()
            messages.index_add_(0, rel_dst, transformed)
        if self.bias is not None:
            messages += self.bias
        return messages


class RelationalGraphEncoder(nn.Module):
    """Stacks relational convolutions to produce graph-level embedding."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        input_dim = config.input_dim
        for _ in range(config.num_layers):
            layer = RelGraphConvLayer(
                in_dim=input_dim,
                out_dim=config.hidden_dim,
                num_relations=config.num_relations,
            )
            layers.append(layer)
            input_dim = config.hidden_dim
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = getattr(nn.functional, config.activation, nn.functional.relu)
        self.readout = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = batch.node_features
        for layer in self.layers:
            layer.num_relations = batch.num_relations  # ensure runtime value
            x = layer(batch, x)
            x = self.activation(x)
            x = self.dropout(x)
        graph_embedding = x.mean(dim=0)
        return self.readout(graph_embedding)

