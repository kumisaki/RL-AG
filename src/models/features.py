"""Feature engineering utilities for graph-based policy learning."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch

from data.models import EntityAlphabet, PolicyDefinition
from env.apt_env import AttackObservation

from .encoders import GraphBatch


def build_relation_vocab(policies: Iterable[PolicyDefinition]) -> Dict[str, int]:
    relations = sorted(
        {triple.relation for policy in policies for triple in policy.triples}
    )
    return {relation: idx for idx, relation in enumerate(relations)}


def encode_observation(
    observation: AttackObservation,
    relation_vocab: Dict[str, int],
) -> GraphBatch:
    nodes = list(observation.provenance.iter_nodes())
    node_index = {node.node_id: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    if num_nodes == 0:
        # bootstrap with a single dummy node to avoid empty tensors
        node_features = torch.zeros((1, len(EntityAlphabet)), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_types = torch.zeros((0,), dtype=torch.long)
        return GraphBatch(
            node_features=node_features,
            edge_index=edge_index,
            edge_types=edge_types,
            num_relations=len(relation_vocab),
        )

    node_features = torch.zeros((num_nodes, len(EntityAlphabet)), dtype=torch.float32)
    for node in nodes:
        idx = node_index[node.node_id]
        try:
            type_idx = EntityAlphabet.index(node.entity_type)
        except ValueError:
            type_idx = 0
        node_features[idx, type_idx] = 1.0

    edges = list(observation.provenance.iter_edges())
    if edges:
        edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
        edge_types = torch.zeros((len(edges),), dtype=torch.long)
        for pos, edge in enumerate(edges):
            edge_index[0, pos] = node_index[edge.subject]
            edge_index[1, pos] = node_index[edge.obj]
            edge_types[pos] = relation_vocab.get(edge.relation, 0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_types = torch.zeros((0,), dtype=torch.long)

    return GraphBatch(
        node_features=node_features,
        edge_index=edge_index,
        edge_types=edge_types,
        num_relations=len(relation_vocab),
    )

