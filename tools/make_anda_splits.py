#!/usr/bin/env python3
"""
Simple CIFAR10 label-skew splitter (Dirichlet) -> writes:
data/cur_datasets/client_{i}.npy with:
  - train_features: uint8 array [N, 3, 32, 32]
  - train_labels:   int64 array [N]
"""

import argparse
from pathlib import Path
import numpy as np
from torchvision.datasets import CIFAR10

def level_to_alpha(level: str) -> float:
    level = level.lower()
    if level == "low":
        return 10.0
    if level == "medium":
        return 0.5
    if level == "high":
        return 0.1
    if level == "very_high":
        return 0.03
    raise ValueError("non_iid_level must be one of: low, medium, high, very_high")

def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    idx_by_class = [np.where(y == k)[0].tolist() for k in range(n_classes)]
    for k in range(n_classes):
        rng.shuffle(idx_by_class[k])

    client_indices = [[] for _ in range(n_clients)]
    for k in range(n_classes):
        idx_k = idx_by_class[k]
        if len(idx_k) == 0:
            continue
        proportions = rng.dirichlet(alpha=np.full(n_clients, alpha))
        counts = (proportions * len(idx_k)).astype(int)
        diff = len(idx_k) - counts.sum()
        for i in range(abs(diff)):
            counts[i % n_clients] += 1 if diff > 0 else -1

        start = 0
        for cid in range(n_clients):
            c = counts[cid]
            if c > 0:
                client_indices[cid].extend(idx_k[start:start+c])
                start += c

    for cid in range(n_clients):
        rng.shuffle(client_indices[cid])
    return client_indices

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/cur_datasets")
    ap.add_argument("--num_clients", type=int, default=15)
    ap.add_argument("--non_iid_level", type=str, default="medium")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha = level_to_alpha(args.non_iid_level)

    ds = CIFAR10(root="data", train=True, download=args.download, transform=None)
    X = ds.data  # uint8 [N, 32, 32, 3]
    y = np.array(ds.targets, dtype=np.int64)

    client_idx = dirichlet_partition(y, args.num_clients, alpha=alpha, seed=args.seed)

    print(f"[splitter] Writing {args.num_clients} clients to {out_dir}")
    for cid in range(args.num_clients):
        idx = np.array(client_idx[cid], dtype=np.int64)
        Xc = X[idx]  # [Nc, 32, 32, 3]
        yc = y[idx]

        Xc = np.transpose(Xc, (0, 3, 1, 2)).copy()  # -> [Nc, 3, 32, 32]

        payload = {"train_features": Xc.astype(np.uint8), "train_labels": yc.astype(np.int64)}
        np.save(out_dir / f"client_{cid}.npy", payload)

        binc = np.bincount(yc, minlength=10)
        print(f"  client_{cid}: N={len(yc)} | label_counts={binc.tolist()}")

    print("[splitter] Done.")

if __name__ == "__main__":
    main()
