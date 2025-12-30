import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Ensure reproducibility across torch, numpy, and python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    """Creates a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def check_gpu():
    """Returns best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_trainable_params(model):
    """Returns count of all trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_cluster_stats(cluster_map: dict) -> str:
    """Pretty cluster summary."""
    return " | ".join([f"C{cid}:{len(members)} clients" for cid, members in cluster_map.items()])
