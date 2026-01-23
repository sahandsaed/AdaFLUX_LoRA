#!/usr/bin/env python3
"""
Sequential AdaFLUX-LoRA Simulation (no Flower networking)
- Uses the same public functions from src/client.py
- No _init_model(), no missing collect_descriptor_vector
- Loads data from data/cur_datasets/client_{i}.npy

Run:
  python src/fedseq_adaflux_lora.py --rounds 10 --fit_clients 3 --eval_clients 6
"""

import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import os, sys, gc, argparse
from types import SimpleNamespace
from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.cluster import DBSCAN

import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import config as cfg
import utils


from client import FlowerClient, build_vit_adalora, collect_lora_keys, vit_descriptor_vector
from logging_utils import TensorboardLogger
from visualize_clusters import plot_flux_embeddings



def aggregate_cluster_updates(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    total = sum(n for _, n in results)
    stacked = [[arr * n for arr in params] for params, n in results]
    return [reduce(np.add, layer) / max(1, total) for layer in zip(*stacked)]


class FLUXClusterManager:
    def __init__(self, interval=3):
        self.store: Dict[int, np.ndarray] = {}
        self.cluster_map: Dict[int, List[int]] = {}
        self.client_to_cluster: Dict[int, int] = {}
        self.interval = interval

    def update(self, cid: int, desc_vec: np.ndarray):
        self.store[cid] = np.asarray(desc_vec, dtype=np.float32)

    def recluster(self, rnd: int) -> bool:
        if rnd % self.interval != 0 or len(self.store) < 2:
            return False

        cids = sorted(self.store.keys())
        X = np.stack([self.store[c] for c in cids])
        d = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
        eps = max(np.percentile(d, 25), 1e-6)
        labels = DBSCAN(eps=eps, min_samples=2).fit(X).labels_

        self.cluster_map.clear()
        self.client_to_cluster.clear()
        for i, cid in enumerate(cids):
            cl = int(labels[i])
            self.cluster_map.setdefault(cl, []).append(cid)
            self.client_to_cluster[cid] = cl

        print(f"[fedseq] Updated clusters -> {self.summary()}")
        return True

    def get_cluster(self, cid: int) -> int:
        return self.client_to_cluster.get(cid, -1)

    def summary(self):
        return {int(c): len(v) for c, v in self.cluster_map.items()}


def get_state(model, keys):
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy() for k in keys]


def set_state(model, keys, tensors):
    with torch.no_grad():
        sd = model.state_dict()
        for k, v in zip(keys, tensors):
            sd[k].copy_(torch.tensor(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=int(getattr(cfg, "n_rounds", 10)))
    ap.add_argument("--fit_clients", type=int, default=3)
    ap.add_argument("--eval_clients", type=int, default=6)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_ckpt_each_round", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    utils.set_seed(int(getattr(cfg, "random_seed", 42)) + int(args.fold))

    n_clients = int(getattr(cfg, "n_clients", 15))
    print(f"\n🚀 Starting FedSeq AdaFLUX-LoRA | clients={n_clients} device={device}\n")

    logger = TensorboardLogger(run_name="AdaFLUX-LoRA-FedSeq")

    # Build clients
    clients: Dict[int, FlowerClient] = {}
    for cid in range(n_clients):
        dummy_args = SimpleNamespace(id=cid, fold=args.fold, exchange_head=False)
        model = build_vit_adalora(num_classes=int(getattr(cfg, "n_classes", 10))).to(device)
        clients[cid] = FlowerClient(model=model, device=device, task="vision", tokenizer=None, args=dummy_args)

    lora_keys = collect_lora_keys(clients[0].model.state_dict().keys(), include_head=False)
    global_params = get_state(clients[0].model, lora_keys)

    cluster_mgr = FLUXClusterManager(interval=int(getattr(cfg, "cluster_update_interval", 3)))
    history = {"loss": [], "acc": []}

    for rnd in range(1, args.rounds + 1):
        print(f"\n====== ROUND {rnd} ======")
        fit_ids = np.random.choice(n_clients, size=min(args.fit_clients, n_clients), replace=False)

        cluster_updates: Dict[int, List[Tuple[List[np.ndarray], int]]] = {}

        for cid in fit_ids:
            clients[cid].set_parameters(global_params)
            params, count, metrics = clients[cid].fit(global_params, {"current_round": rnd, "local_epochs": int(getattr(cfg, "local_epochs", 1))})

            # descriptor
            train_loader = clients[cid].load_current_data(rnd, train=True)
            desc_vec = vit_descriptor_vector(clients[cid].model, train_loader, device, max_batches=int(getattr(cfg, "descriptor_max_batches", 2)))
            cluster_mgr.update(cid, desc_vec)

            cl = cluster_mgr.get_cluster(cid)
            cluster_updates.setdefault(cl, []).append((params, count))

        cluster_mgr.recluster(rnd)

        # aggregate per cluster
        new_params_by_cluster = {cl: aggregate_cluster_updates(upds) for cl, upds in cluster_updates.items()}
        fallback = list(new_params_by_cluster.values())[0]
        global_params = fallback

        # apply cluster params to each client (local simulation)
        for cid in range(n_clients):
            cl = cluster_mgr.get_cluster(cid)
            set_state(clients[cid].model, lora_keys, new_params_by_cluster.get(cl, fallback))

        # evaluate
        eval_ids = np.random.choice(n_clients, size=min(args.eval_clients, n_clients), replace=False)
        losses, accs = [], []
        for cid in eval_ids:
            loss, _, m = clients[cid].evaluate(global_params, {"current_round": rnd})
            losses.append(loss)
            accs.append(m.get("accuracy", 0.0))

        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_acc = float(np.mean(accs)) if accs else 0.0
        history["loss"].append(avg_loss)
        history["acc"].append(avg_acc)

        print(f"📈 Round {rnd} | Loss={avg_loss:.4f} | Acc={avg_acc:.4f}")
        logger.log_round_metrics(rnd, avg_loss, avg_acc, cluster_mgr.summary())

        if args.save_ckpt_each_round:
            os.makedirs("checkpoints_local", exist_ok=True)
            torch.save({"keys": lora_keys, "tensors": [torch.tensor(x) for x in global_params]},
                       f"checkpoints_local/round_{rnd}_global.pt")

        torch.cuda.empty_cache()
        gc.collect()

    print("\n🎯 Finished Training")
    print("Cluster Summary:", cluster_mgr.summary())

    # plot clusters
    plot_flux_embeddings(cluster_mgr.store, cluster_mgr.client_to_cluster, out_path="flux_clusters_fedseq.png")

    os.makedirs("results_local", exist_ok=True)
    np.save("results_local/history.npy", history)
    logger.close()
    print("💾 Logs saved and logger closed.")


if __name__ == "__main__":
    main()
