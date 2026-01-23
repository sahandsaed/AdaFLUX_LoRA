import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import sys, json, argparse
from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import torch
from sklearn.cluster import DBSCAN

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import config as cfg
import utils

from client import build_vit_adalora, collect_lora_keys


def aggregate_weights(weighted_updates):
    total = sum(n for _, n in weighted_updates)
    stacked = [[layer * n for layer in client] for client, n in weighted_updates]
    return [reduce(np.add, layers) / max(1, total) for layers in zip(*stacked)]


class FLUXClusterManager:
    def __init__(self, recluster_every=3):
        self.descriptors: Dict[int, np.ndarray] = {}
        self.cluster_map: Dict[int, List[int]] = {}
        self.client_cluster: Dict[int, int] = {}
        self.recluster_every = recluster_every

    def update_descriptor(self, cid: int, desc_vec: List[float]):
        self.descriptors[cid] = np.asarray(desc_vec, dtype=np.float32)

    def recluster(self, rnd: int) -> bool:
        if rnd % self.recluster_every != 0:
            return False
        if len(self.descriptors) < 2:
            return False

        cids = sorted(self.descriptors.keys())
        X = np.stack([self.descriptors[c] for c in cids])
        d = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
        eps = max(np.percentile(d, 25), 1e-6)
        labels = DBSCAN(eps=eps, min_samples=2).fit(X).labels_

        self.cluster_map.clear()
        self.client_cluster.clear()
        for i, cid in enumerate(cids):
            cl = int(labels[i])
            self.cluster_map.setdefault(cl, []).append(cid)
            self.client_cluster[cid] = cl

        print(f"[server] Reclustered @ round {rnd}: {self.summary()}")
        return True

    def get_cluster(self, cid: int) -> int:
        return self.client_cluster.get(cid, -1)

    def summary(self):
        return {int(k): len(v) for k, v in self.cluster_map.items()}


class AdaFLUXClusterStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, lora_keys, cluster_mgr: FLUXClusterManager, exp_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.lora_keys = lora_keys
        self.cluster_mgr = cluster_mgr
        self.exp_dir = exp_dir

    def initialize_parameters(self, client_manager):
        tensors = [self.model.state_dict()[k].detach().cpu().numpy() for k in self.lora_keys]
        return ndarrays_to_parameters(tensors)

    def aggregate_fit(self, rnd, results, failures):
        for proxy, fit_res in results:
            m = fit_res.metrics or {}
            if "descriptor" in m:
                cid = int(proxy.cid)
                self.cluster_mgr.update_descriptor(cid, m["descriptor"])

        self.cluster_mgr.recluster(rnd)

        clusters: Dict[int, List[Tuple[List[np.ndarray], int]]] = {}
        for proxy, fit_res in results:
            cid = int(proxy.cid)
            cl = self.cluster_mgr.get_cluster(cid)
            params = parameters_to_ndarrays(fit_res.parameters)
            clusters.setdefault(cl, []).append((params, fit_res.num_examples))

        aggregated_per_cluster = {}
        for cl, updates in clusters.items():
            merged = aggregate_weights(updates)
            aggregated_per_cluster[cl] = merged

            ckpt_dir = Path("checkpoints") / self.exp_dir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"keys": self.lora_keys, "tensors": [torch.tensor(x) for x in merged]},
                ckpt_dir / f"cluster_{cl}_round_{rnd}.pt",
            )

        new_params = aggregated_per_cluster.get(-1) or list(aggregated_per_cluster.values())[0]
        metrics = {"clusters": json.dumps(self.cluster_mgr.summary())}
        print(f"[server] Round {rnd} clusters: {self.cluster_mgr.summary()}")
        return ndarrays_to_parameters(new_params), metrics


def fit_cfg(rnd: int):
    return {"current_round": rnd, "local_epochs": int(cfg.local_epochs)}


def eval_weighted(mets):
    total = sum(n for n, _ in mets)
    acc = sum(n * m.get("accuracy", 0.0) for n, m in mets)
    return {"accuracy": float(acc / max(1, total))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    run_dir = "run_default"
    utils.set_seed(int(cfg.random_seed) + int(args.fold))

    device = utils.check_gpu()
    model = build_vit_adalora(num_classes=int(cfg.n_classes)).to(device)
    lora_keys = collect_lora_keys(model.state_dict().keys(), include_head=False)

    cluster_mgr = FLUXClusterManager(recluster_every=int(cfg.cluster_update_interval))

    strategy = AdaFLUXClusterStrategy(
        model=model,
        lora_keys=lora_keys,
        cluster_mgr=cluster_mgr,
        exp_dir=run_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=int(cfg.n_clients),
        on_fit_config_fn=fit_cfg,
        evaluate_metrics_aggregation_fn=eval_weighted,
    )

    fl.server.start_server(
        server_address=f"{cfg.ip}:{cfg.port}",
        config=fl.server.ServerConfig(num_rounds=int(cfg.n_rounds)),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
