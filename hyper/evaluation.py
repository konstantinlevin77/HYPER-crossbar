"""Evaluation helpers that are independent from sampling and model tasks."""

from collections import defaultdict

import torch
from torch import distributed as dist


# RULE datasets have a single relation and a fixed positional schema.
RULE_NODE_TYPES = {
    "HYPERRULE1": ("disease", "gene", "drug", "pathway"),
    "HYPERRULE2": ("disease", "gene", "drug", "pathway"),
    "HYPER-RULE1": ("disease", "gene", "drug", "pathway"),
    "HYPER-RULE2": ("disease", "gene", "drug", "pathway"),
    "HYPERRULE1_STF": ("disease", "gene", "drug", "pathway"),
    "HYPER-RULE1_STF": ("disease", "gene", "drug", "pathway"),
    "HYPERRULE2_STF": ("disease", "gene", "drug", "pathway"),
    "HYPER-RULE2_STF": ("disease", "gene", "drug", "pathway"),
}


def node_types_for_dataset(dataset_name):
    """Return ordered node-type names for a dataset, or ``None`` if unknown."""
    return RULE_NODE_TYPES.get(str(dataset_name).upper())


def new_position_rankings():
    """Create an accumulator of typed ranks keyed by hyperedge position."""
    return defaultdict(list)


def add_position_rankings(accumulator, position, rankings):
    """Record the typed ranks produced while corrupting one position."""
    if rankings is not None and rankings.numel():
        accumulator[position].append(rankings)


def node_specific_typed_metrics(position_rankings, node_types, device, distributed=True):
    """Compute MRR and Hits from typed ranks, grouped by fixed node position.

    Sufficient statistics are reduced instead of gathering every rank, keeping
    this inexpensive for the large RULE evaluation sets.
    """
    if not node_types:
        return {}

    metrics = {}
    for position, node_type in enumerate(node_types):
        rank_parts = position_rankings.get(position, ())
        rankings = torch.cat(rank_parts) if rank_parts else torch.empty(0, device=device)
        rankings = rankings.to(device=device, dtype=torch.float64)
        stats = torch.tensor(
            [
                rankings.numel(),
                (1 / rankings).sum().item() if rankings.numel() else 0,
                (rankings <= 1).sum().item(),
                (rankings <= 3).sum().item(),
                (rankings <= 10).sum().item(),
            ],
            dtype=torch.float64,
            device=device,
        )
        if distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        count = int(stats[0].item())
        if count == 0:
            continue
        prefix = f"typed/{node_type}"
        metrics[f"{prefix}/count"] = count
        metrics[f"{prefix}/mrr"] = stats[1] / count
        metrics[f"{prefix}/hits@1"] = stats[2] / count
        metrics[f"{prefix}/hits@3"] = stats[3] / count
        metrics[f"{prefix}/hits@10"] = stats[4] / count
    return metrics
