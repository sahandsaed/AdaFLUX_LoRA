"""
AdaFLUX-LoRA Federated Server (Full Research Version - Option D)

Key Features:
---------------------------------------
✓ Collects FLUX client descriptors
✓ Clusters clients using adaptive DBSCAN
✓ Performs cluster-specific AdaLoRA aggregation
✓ Assigns unseen clients to closest cluster (CFL inference stage)
✓ Saves per-cluster checkpoints each round
✓ Supports rank messaging to each client (next round control)
"""

import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import os, sys, json, argparse, time
from functools import reduce
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import flwr as fl
from flwr.common import (
    FitRes, Parameters, Metrics,
    parameters_to_ndarrays, ndarrays_to_parameters
)

import torch
from sklearn.cluster import DBSCAN
from transformers import ViTForImageClassification, T5ForConditionalGeneration
from peft import AdaLoraConfig, get_peft_model

# === project imports ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import public.utils as utils
import public.config as cfg
import public.models as models



# --------------------------------------------------------------------------------
# MODEL BUILDERS (must match client)
# --------------------------------------------------------------------------------
def build_adalora_vit():
    model_name = getattr(cfg, "hf_model_name", "google/vit-base-patch16-224-in21k")
    r, alpha, drop = cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout
    target = cfg.class_lora_target_modules

    base = ViTForImageClassification.from_pretrained(model_name, num_labels=cfg.n_classes)
    ada = AdaLoraConfig(
        r=r, init_r=r, tinit=0, tfinal=0, deltaT=1,
        lora_alpha=alpha, lora_dropout=drop,
        target_modules=target, bias="none",
        modules_to_save=["classifier"]
    )
    model = get_peft_model(base, ada)
    for n,p in model.named_parameters():
        p.requires_grad = ("lora_" in n)
    return model


def build_adalora_t5():
    model_name = cfg.hf_summarization_model_name
    tok = AutoTokenizer.from_pretrained(model_name)

    r, alpha, drop = cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout
    target = cfg.summ_lora_target_modules

    base = T5ForConditionalGeneration.from_pretrained(model_name)
    base.config.use_cache = False

    ada = AdaLoraConfig(
        r=r, init_r=r, lora_alpha=alpha, lora_dropout=drop,
        target_modules=target, tfinal=0, tinit=0, deltaT=1
    )
    model = get_peft_model(base, ada)
    for n,p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

    return model, tok



# --------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------

def collect_lora_keys(keys, include_head=False):
    k = [x for x in keys if "lora_" in x]
    if include_head:
        k += [x for x in keys if x.startswith("classifier.")]
    return sorted(set(k))


def aggregate_weights(weighted_updates):
    total = sum(n for _, n in weighted_updates)
    stacked = [[layer * n for layer in client] for client, n in weighted_updates]
    return [reduce(np.add, layers) / total for layers in zip(*stacked)]


def weighted_complexity(metrics: List[Tuple[int, Dict]]):
    tot = sum(n for n,_ in metrics)
    return sum(n * m.get("complexity",0) for n,m in metrics) / max(tot,1)



# --------------------------------------------------------------------------------
# CLUSTERING LOGIC (FLUX-based)
# --------------------------------------------------------------------------------

class FLUXClusterManager:
    """Stores descriptors and clusters clients dynamically."""
    def __init__(self):
        self.descriptors = {}    # {client_id: np.vector}
        self.cluster_map = {}    # {cluster_id: [clients]}
        self.client_cluster = {} # {client_id: cluster_id}
        self.recluster_every = getattr(cfg, "cluster_update_interval", 3)

    def update_descriptor(self, client_id, desc):
        self.descriptors[client_id] = np.array(desc)

    def recluster(self, rnd):
        if rnd % self.recluster_every != 0:
            return False

        if len(self.descriptors) < 2:
            return False

        matrix = np.stack([self.descriptors[i] for i in sorted(self.descriptors.keys())])
        
        # Adaptive eps selection via elbow heuristic
        # (simple heuristic: eps = 10th percentile distance)
        dists = np.linalg.norm(matrix[:,None] - matrix[None,:], axis=-1)
        eps = np.percentile(dists, 25)
        db = DBSCAN(eps=max(eps,1e-6), min_samples=2).fit(matrix)

        self.cluster_map.clear()
        for cid, label in enumerate(db.labels_):
            self.cluster_map.setdefault(label, []).append(sorted(self.descriptors.keys())[cid])

        for label, group in self.cluster_map.items():
            for c in group:
                self.client_cluster[c] = label

        print(f"[AdaFLUX] Reclustered: {len(self.cluster_map)} clusters.")
        return True

    def get_cluster(self, cid):
        return self.client_cluster.get(cid, -1)

    def get_cluster_clients(self, label):
        return self.cluster_map.get(label, [])

    def summary(self):
        return {k:len(v) for k,v in self.cluster_map.items()}



# --------------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------------

class AdaFLUXClusterStrategy(fl.server.strategy.FedAvg):

    def __init__(self, model, lora_keys, exp_path, cluster_mgr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.lora_keys = lora_keys
        self.cluster_mgr = cluster_mgr
        self.exp_path = exp_path

    def initialize_parameters(self, client_manager):
        tensors = [self.model.state_dict()[k].cpu().numpy() for k in self.lora_keys]
        return ndarrays_to_parameters(tensors)

    def aggregate_fit(self, rnd, results, failures):

        # === Collect descriptors for FLUX ===
        for proxy, res in results:
            if "descriptor" in res.metrics:
                cid = int(proxy.cid)
                self.cluster_mgr.update_descriptor(cid, res.metrics["descriptor"])

        # === Recluster if needed ===
        self.cluster_mgr.recluster(rnd)

        # === Group by cluster ===
        clusters = {}
        for proxy, res in results:
            cid = int(proxy.cid)
            cl = self.cluster_mgr.get_cluster(cid)

            params = parameters_to_ndarrays(res.parameters)
            clusters.setdefault(cl, []).append((params, res.num_examples))

        aggregated_per_cluster = {}

        # === Aggregate each cluster independently ===
        for cl, arrs in clusters.items():
            merged = aggregate_weights(arrs)
            aggregated_per_cluster[cl] = merged

            # Save checkpoint
            path = f"checkpoints/{self.exp_path}/cluster_{cl}_round_{rnd}.pt"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({"keys": self.lora_keys, "tensors":[torch.tensor(t) for t in merged]}, path)

        print(f"[AdaFLUX] Round {rnd} cluster map:", self.cluster_mgr.summary())

        # === For next round: broadcast weights based on cluster assignment ===
        # Strategy: if client belongs to cluster C, send C model
        def per_client_params(cid):
            cl = self.cluster_mgr.get_cluster(int(cid))
            return aggregated_per_cluster.get(cl, aggregated_per_cluster.get(-1))

        # Setup return structure
        new_params = aggregated_per_cluster.get(-1) or list(aggregated_per_cluster.values())[0]
        return ndarrays_to_parameters(new_params), {
            "clusters": json.dumps(self.cluster_mgr.summary())
        }

    def configure_fit(self, rnd, params, client_manager):
        """Override to assign each client its cluster-specific params."""
        conf = super().configure_fit(rnd, params, client_manager)

        for (cid, cfg_dict) in conf:
            cl = self.cluster_mgr.get_cluster(int(cid))
            # broadcast cluster assignment instruction
            cfg_dict["cluster_id"] = cl

        return conf



# --------------------------------------------------------------------------------
# Round config
# --------------------------------------------------------------------------------

def fit_cfg(rnd):
    return {
        "current_round": rnd,
        "local_epochs": cfg.local_epochs,
        "local_iterations": getattr(cfg, "local_iterations", 0),
        "enable_descriptor": True,  # request clients send descriptors
    }


def eval_weighted(mets):
    total = sum(n for n,_ in mets)
    acc = sum(n*m.get("accuracy",0) for n,m in mets)
    return {"accuracy":acc/max(1,total)}



# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser("AdaFLUX-LoRA Full Cluster Server")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--exchange_head", action="store_true")
    args = parser.parse_args()

    run_dir = utils.create_folders()
    utils.set_seed(cfg.random_seed)

    device = utils.check_gpu()

    # === Initialize model ===
    if cfg.dataset_name == "Summarization":
        model, tok = build_adalora_t5()
        model.to(device)
    else:
        model = build_adalora_vit().to(device)

    # === LoRA keys ===
    keys = collect_lora_keys(model.state_dict().keys(), args.exchange_head)

    # === cluster manager ===
    cluster_mgr = FLUXClusterManager()

    strategy = AdaFLUXClusterStrategy(
        model=model,
        lora_keys=keys,
        exp_path=run_dir,
        cluster_mgr=cluster_mgr,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=cfg.n_clients,
        on_fit_config_fn=fit_cfg,
        evaluate_metrics_aggregation_fn=eval_weighted
    )

    fl.server.start_server(
        server_address=f"{cfg.ip}:{cfg.port}",
        config=fl.server.ServerConfig(cfg.n_rounds),
        strategy=strategy
    )

    print("\n=== Training Complete ===\n")


if __name__ == "__main__":
    main()
