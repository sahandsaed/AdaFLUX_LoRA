"""
AdaFLUX-LoRA Federated Learning Server

- Aggregates only AdaLoRA parameters from clients
- Receives client descriptors + complexity scores for FLUX-style clustering
- Saves per-round AdaLoRA checkpoints
- Supports test-time evaluation and best-round model selection
"""

import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import argparse, json, time, os, sys
from functools import reduce
from typing import Dict, List, Tuple, Optional, Union, Iterable
from pathlib import Path

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    Metrics,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import public.utils as utils
import public.config as cfg
import public.models as models

from flwr.server.client_proxy import ClientProxy

from transformers import AutoTokenizer, ViTForImageClassification, T5ForConditionalGeneration
from peft import AdaLoraConfig, get_peft_model


ROOT_DIR = Path(parent_dir)
DATA_ROOT = ROOT_DIR / "data" / "cur_datasets"
IS_SUMMARIZATION = cfg.dataset_name == "Summarization"



# ----------------------------------------------------------------------
# Summarization utilities (kept identical for compatibility)
# ----------------------------------------------------------------------
def summarization_scenario_from_non_iid(non_iid: str) -> str:
    mapping = {
        "feature_skew_strict": "PX",
        "feature_skew": "PX",
        "label_skew_strict": "PY",
        "label_skew": "PY",
        "label_condition_skew": "PY_given_X",
        "feature_condition_skew": "PX_given_Y",
        "Px": "PX",
        "Py": "PY",
        "Py_x": "PY_given_X",
        "Px_y": "PX_given_Y",
        "PX": "PX",
        "PY": "PY",
        "PY_given_X": "PY_given_X",
        "PX_given_Y": "PX_given_Y",
    }
    return mapping.get(non_iid, non_iid)


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return [json.loads(l) for l in path.open("r", encoding="utf-8")]


class SummarizationExampleDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def build_summarization_collate(tokenizer, max_input_len, max_target_len):
    pad = tokenizer.pad_token_id

    def _coll(batch):
        inputs = tokenizer([b["U"] for b in batch], padding=True, truncation=True,
                           max_length=max_input_len, return_tensors="pt")

        with tokenizer.as_target_tokenizer():
            labels = tokenizer([b["Y"] for b in batch], padding=True, truncation=True,
                               max_length=max_target_len, return_tensors="pt")

        l = labels["input_ids"]
        l[l == pad] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": l,
            "decoder_attention_mask": labels["attention_mask"],
        }, [b["Y"] for b in batch]

    return _coll



# ----------------------------------------------------------------------
# AdaLoRA model builders (aligned with client)
# ----------------------------------------------------------------------
def build_adalora_vit():
    model_name = getattr(cfg, "hf_model_name", "google/vit-base-patch16-224-in21k")
    r = cfg.lora_r
    alpha = cfg.lora_alpha
    dropout = cfg.lora_dropout
    target = cfg.class_lora_target_modules

    base = ViTForImageClassification.from_pretrained(model_name, num_labels=cfg.n_classes)
    ada = AdaLoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target, bias="none",
        init_r=r, tinit=0, tfinal=0, deltaT=1,
        beta1=0.85, beta2=0.85, orth_reg_weight=0.0,
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base, ada)

    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

    return model


def build_adalora_t5():
    model_name = getattr(cfg, "hf_summarization_model_name", "google-t5/t5-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    r = cfg.lora_r
    alpha = cfg.lora_alpha
    dropout = cfg.lora_dropout
    target = cfg.summ_lora_target_modules

    base = T5ForConditionalGeneration.from_pretrained(model_name)
    base.config.use_cache = False

    ada = AdaLoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target, bias="none", init_r=r,
        tinit=0, tfinal=0, deltaT=1
    )
    model = get_peft_model(base, ada)

    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

    return model, tokenizer



# ----------------------------------------------------------------------
# Parameter utilities
# ----------------------------------------------------------------------
def collect_lora_keys(sd_keys, include_head=False):
    k = [p for p in sd_keys if "lora_" in p]
    if include_head:
        k += [p for p in sd_keys if p.startswith("classifier.")]
    return sorted(set(k))


def set_named_(model, keys, tensors):
    sd = model.state_dict()
    with torch.no_grad():
        for k, arr in zip(keys, tensors):
            sd[k].copy_(torch.tensor(arr))



# ----------------------------------------------------------------------
# Aggregation + AdaFLUX metadata handling
# ----------------------------------------------------------------------
def aggregate_lora(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    total = sum(n for _, n in results)
    weighted = [[layer * n for layer in arrs] for arrs, n in results]
    return [reduce(np.add, layers) / total for layers in zip(*weighted)]


def aggregate_complexity(metrics: List[Tuple[int, Metrics]]) -> float:
    """Compute weighted mean client complexity (for logging and clustering telemetry)."""
    weighted = [num * m.get("complexity", 0.0) for num, m in metrics]
    total = sum(num for num, _ in metrics)
    return float(sum(weighted) / max(1, total))



# ----------------------------------------------------------------------
# Strategy with checkpoint saving
# ----------------------------------------------------------------------
class AdaFLUXStrategy(fl.server.strategy.FedAvg):
    """
    Custom FL strategy supporting:

      - AdaLoRA weight aggregation
      - Complexity score telemetry
      - Per-round checkpoints
    """

    def __init__(self, model, lora_keys, path, args, *x, **kw):
        super().__init__(*x, **kw)
        self.model = model
        self.lora_keys = lora_keys
        self.path = path
        self.args = args

    def initialize_parameters(self, client_manager):
        tensors = [self.model.state_dict()[k].cpu().numpy() for k in self.lora_keys]
        return ndarrays_to_parameters(tensors)

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # Extract updated params
        param_results = [
            (parameters_to_ndarrays(r.parameters), r.num_examples)
            for _, r in results
        ]
        aggregated = aggregate_lora(param_results)
        parameters = ndarrays_to_parameters(aggregated)

        # --- collect complexity metadata ---
        fit_metrics = [(r.num_examples, r.metrics) for _, r in results]
        avg_complex = aggregate_complexity(fit_metrics)

        print(f"[AdaFLUX] Round {rnd} aggregated complexity = {avg_complex:.4f}")

        # --- apply update to server model for checkpointing ---
        set_named_(self.model, self.lora_keys, aggregated)

        # --- checkpoint ---
        os.makedirs(f"checkpoints/{self.path}", exist_ok=True)
        ckpt = {"keys": self.lora_keys, "tensors": [torch.tensor(t) for t in aggregated]}
        torch.save(ckpt, f"checkpoints/{self.path}/rd_{rnd}.pt")

        return parameters, {"complexity": avg_complex}



# ----------------------------------------------------------------------
# Round config
# ----------------------------------------------------------------------
def fit_cfg(rnd):
    return {
        "current_round": rnd,
        "local_epochs": cfg.local_epochs,
        "local_iterations": getattr(cfg, "local_iterations", 0),
    }


def agg_eval(metrics):
    acc = [num*m.get("accuracy", 0.0) for num, m in metrics]
    total = sum(num for num, _ in metrics)
    return {"accuracy": float(sum(acc)/max(1,total))}



# ----------------------------------------------------------------------
# Evaluation utilities
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_vit(model, device, loader):
    prep = lambda x: {"pixel_values": x.to(device)}
    total, correct, loss_total = 0, 0, 0.0
    model.eval()

    for xb, yb in loader:
        xb = xb.to(device)
        out = model(**prep(xb), labels=yb.to(device))
        loss_total += float(out.loss)
        preds = out.logits.argmax(-1).cpu()
        correct += (preds == yb).sum().item()
        total += yb.numel()

    return loss_total/len(loader), correct/max(1,total)



# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser("AdaFLUX-LoRA Server")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--exchange_head", action="store_true")
    parser.add_argument("--non_iid_type", default=cfg.non_iid_type)
    args = parser.parse_args()

    utils.set_seed(cfg.random_seed)

    # GPU
    device = utils.check_gpu()

    # Experiment path
    run_dir = utils.create_folders()

    # Build model
    if IS_SUMMARIZATION:
        model, tokenizer = build_adalora_t5()
        model.to(device)
    else:
        model = build_adalora_vit().to(device)
        tokenizer = None

    # LoRA keys
    keys = collect_lora_keys(model.state_dict().keys(), args.exchange_head)

    # Strategy
    strategy = AdaFLUXStrategy(
        model=model, lora_keys=keys, path=run_dir, args=args,
        fraction_fit=1.0,
        min_fit_clients=cfg.min_fit_clients if hasattr(cfg,"min_fit_clients") else 3,
        min_available_clients=cfg.n_clients,
        fraction_evaluate=1.0,
        on_fit_config_fn=fit_cfg,
        evaluate_metrics_aggregation_fn=agg_eval,
    )

    # Start server
    fl.server.start_server(
        server_address=f"{cfg.ip}:{cfg.port}",
        config=fl.server.ServerConfig(cfg.n_rounds),
        strategy=strategy,
    )

    print("\n[Server] Finished training.\n")


if __name__ == "__main__":
    main()
