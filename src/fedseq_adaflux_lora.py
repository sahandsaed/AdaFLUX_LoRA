#!/usr/bin/env python3
"""
AdaFLUX-LoRA Sequential Federated Simulation
--------------------------------------------

Simulates the behavior of the AdaFLUX-LoRA server/client system locally,
without running Flower networking.

Features:
    ✓ FLUX descriptor collection
    ✓ Dynamic clustering using DBSCAN
    ✓ Cluster-level aggregation (CFL-style)
    ✓ AdaLoRA rank adaptation passthrough
    ✓ Weighted FedAvg inside clusters
    ✓ Same training/eval pipeline as server & client

Run example:
    python fedseq_adaflux_lora.py --rounds 10 --fit_clients 3 --eval_clients 6
"""

import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import os, sys, gc, json, time, copy, argparse
from types import SimpleNamespace
from functools import reduce
from typing import Dict, List, Tuple
from pathlib import Path

import torch
from sklearn.cluster import DBSCAN
import flwr as fl

# ------------------------------
# CHANGE 1: Load TensorBoard logger
# ------------------------------
from logging_utils import TensorboardLogger
logger = TensorboardLogger(run_name="AdaFLUX-LoRA")

# Add project path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

# Import shared modules
import public.config as cfg
import public.utils as utils
from client_adaflux_lora import FlowerClient, collect_descriptor_vector
from public.utils import plot_loss_and_accuracy
from visualize_clusters import plot_flux_embeddings      # <-- CHANGE 4
from router import FLUXRouter                             # <-- CHANGE 5


# =========================
# Global helpers
# =========================

def collect_lora_keys(keys):
    return sorted([k for k in keys if "lora_" in k])


def get_state(model, keys):
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy() for k in keys]


def set_state(model, keys, tensors):
    with torch.no_grad():
        sd = model.state_dict()
        for k, v in zip(keys, tensors):
            sd[k].copy_(torch.tensor(v))


def aggregate_cluster_updates(results):
    """FedAvg: [(param_list, n), ...] -> averaged param_list"""
    total = sum(n for _, n in results)
    weighted = [[arr * n for arr in params] for params, n in results]
    return [reduce(np.add, layer) / total for layer in zip(*weighted)]


# =========================
# Cluster manager
# =========================

class FLUXClusterManager:
    def __init__(self, interval=3):
        self.store = {}
        self.cluster_map = {}
        self.client_to_cluster = {}
        self.interval = interval

    def update(self, cid, desc):
        self.store[cid] = desc

    def recluster(self, rnd):
        if rnd % self.interval != 0 or len(self.store) < 2:
            return False

        X = np.stack([self.store[c] for c in sorted(self.store.keys())])

        d = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
        eps = max(np.percentile(d, 25), 1e-6)
        labels = DBSCAN(eps=eps, min_samples=2).fit(X).labels_

        self.cluster_map = {}
        for i, cid in enumerate(sorted(self.store.keys())):
            cl = labels[i]
            self.cluster_map.setdefault(cl, []).append(cid)
            self.client_to_cluster[cid] = cl

        print(f"[AdaFLUX] Updated clusters -> {self.summary()}")
        return True

    def get_cluster(self, cid):
        return self.client_to_cluster.get(cid, -1)

    def members(self, cluster_id):
        return self.cluster_map.get(cluster_id, [])

    def summary(self):
        return {c: len(v) for c, v in self.cluster_map.items()}


# =========================
# Local Simulation
# =========================

def main():

    parser = argparse.ArgumentParser("Sequential AdaFLUX-LoRA Training")
    parser.add_argument("--rounds", default=cfg.n_rounds, type=int)
    parser.add_argument("--fit_clients", default=3, type=int)
    parser.add_argument("--eval_clients", default=6, type=int)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_ckpt_each_round", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    utils.set_seed(cfg.random_seed + args.fold)

    print(f"\n🚀 Starting AdaFLUX-LoRA Sequential Simulation | device={device}")

    n_clients = cfg.n_clients
    clients = {}

    for cid in range(n_clients):
        dummy_args = SimpleNamespace(id=cid, fold=args.fold, non_iid_type=cfg.non_iid_type, exchange_head=False)
        model = FlowerClient(model=None, device=device, tokenizer=None, args=dummy_args)._init_model()
        clients[cid] = FlowerClient(model=model, device=device, tokenizer=None, args=dummy_args)

    lora_keys = collect_lora_keys(clients[0].model.state_dict().keys())
    global_params = get_state(clients[0].model, lora_keys)

    cluster_mgr = FLUXClusterManager()
    history = {"loss": [], "acc": []}

    for rnd in range(1, args.rounds + 1):
        print(f"\n====== ROUND {rnd} ======")

        fit_ids = np.random.choice(n_clients, args.fit_clients, replace=False)
        cluster_updates = {}

        for cid in fit_ids:
            clients[cid].set_parameters(global_params)
            params, count, _ = clients[cid].fit(global_params, {"current_round": rnd})

            desc = collect_descriptor_vector(clients[cid])
            cluster_mgr.update(cid, desc)

            cl = cluster_mgr.get_cluster(cid)
            cluster_updates.setdefault(cl, []).append((params, count))

        cluster_mgr.recluster(rnd)

        new_params_by_cluster = {cl: aggregate_cluster_updates(updates) for cl, updates in cluster_updates.items()}
        fallback = list(new_params_by_cluster.values())[0]

        for cid in range(n_clients):
            cl = cluster_mgr.get_cluster(cid)
            set_state(clients[cid].model, lora_keys, new_params_by_cluster.get(cl, fallback))

        global_params = fallback

        eval_ids = np.random.choice(n_clients, args.eval_clients, replace=False)
        losses, accs = [], []
        for cid in eval_ids:
            loss, _, metrics = clients[cid].evaluate(global_params, {"current_round": rnd})
            losses.append(loss)
            accs.append(metrics["accuracy"])

        avg_loss = np.mean(losses)
        avg_acc = np.mean(accs)
        history["loss"].append(avg_loss)
        history["acc"].append(avg_acc)

        print(f"📈 Round {rnd} | Loss={avg_loss:.4f} | Acc={avg_acc:.4f}")

        # ------------------------------
        # CHANGE 2: Log round metrics
        # ------------------------------
        logger.log_round_metrics(rnd, avg_loss, avg_acc, cluster_mgr.summary())

        if args.save_ckpt_each_round:
            os.makedirs("checkpoints_local", exist_ok=True)
            torch.save({"keys": lora_keys, "tensors": [torch.tensor(x) for x in global_params]},
                       f"checkpoints_local/round_{rnd}_global.pt")

    print("\n🎯 Finished Training")
    print("Cluster Summary:", cluster_mgr.summary())

    # ------------------------------
    # CHANGE 3: Plot cluster embeddings visualization
    # ------------------------------
    plot_flux_embeddings(cluster_mgr.store, cluster_mgr.client_to_cluster)

    # ------------------------------
    # CHANGE 4: Test-Time Routing Example
    # ------------------------------
    router = FLUXRouter(cluster_mgr.store, cluster_mgr.client_to_cluster, new_params_by_cluster)
    example_desc = collect_descriptor_vector(clients[0])
    assigned_model, assigned_cluster = router.route(example_desc)
    print(f"🤖 Test-Time Routing → Assigned to cluster {assigned_cluster}")

    os.makedirs("results_local", exist_ok=True)
    np.save("results_local/history.npy", history)

    # ------------------------------
    # CHANGE 5: Close logger
    # ------------------------------
    logger.close()
    print("💾 Logs saved and logger closed.")


if __name__ == "__main__":
    main()
