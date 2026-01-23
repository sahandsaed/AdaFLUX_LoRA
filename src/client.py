import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Dict, List

import flwr as fl
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ---- NEW: make repo root importable ----
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
DATA_ROOT = ROOT_DIR / "data" / "cur_datasets"

# ---- NEW: correct imports (no public/) ----
import config as cfg
import utils

from transformers import ViTForImageClassification
from peft import AdaLoraConfig, get_peft_model

class CombinedDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

IS_SUMMARIZATION = getattr(cfg, "dataset_name", "CIFAR10") == "Summarization"


# ------------------ Descriptor helpers ------------------

@torch.no_grad()
def vit_descriptor_vector(model, loader, device, max_batches: int = 2, k: int = 64) -> np.ndarray:
    """
    Returns a fixed-length vector:
      concat( mean[:k], var[:k], label_hist[:10] ) => length 2k + 10
    """
    model.eval()
    embs = []
    labels = []

    for b_idx, (xb, yb) in enumerate(loader):
        if b_idx >= max_batches:
            break
        xb = xb.to(device, dtype=torch.float32)

        # Ensure [B,3,H,W] float in [0,1]
        if xb.dim() == 3:
            xb = xb.unsqueeze(1)
        if xb.size(1) == 1:
            xb = xb.repeat(1, 3, 1, 1)
        if torch.max(xb) > 1.5:
            xb = xb / 255.0

        size = int(getattr(model.config, "image_size", 224))
        try:
            xb = F.interpolate(xb, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
        except TypeError:
            xb = F.interpolate(xb, size=(size, size), mode="bicubic", align_corners=False)

        feats = model.vit(pixel_values=xb).last_hidden_state  # [B, N, D]
        pooled = feats.mean(dim=1)  # [B, D]
        embs.append(pooled.detach().cpu())
        labels.append(yb.detach().cpu())

    if not embs:
        return np.zeros((2 * k + 10,), dtype=np.float32)

    Z = torch.cat(embs, dim=0)  # [N, D]
    Y = torch.cat(labels, dim=0).to(torch.int64)

    mu = Z.mean(dim=0)
    var = Z.var(dim=0)

    # fixed size slices
    mu_k = mu[:k]
    var_k = var[:k]

    hist = torch.bincount(Y, minlength=10).float()
    hist = hist / hist.sum().clamp_min(1.0)

    vec = torch.cat([mu_k, var_k, hist], dim=0).numpy().astype(np.float32)
    return vec


def compute_complexity_from_vec(vec: np.ndarray, k: int = 64) -> float:
    """
    Simple complexity:
      0.5 * ||var|| + 0.5 * entropy(hist)
    """
    vec = np.asarray(vec, dtype=np.float64)
    var = vec[k:2*k]
    hist = vec[2*k:2*k+10] + 1e-12
    hist = hist / hist.sum()
    entropy = float(-(hist * np.log(hist)).sum())
    feat = float(np.linalg.norm(var))
    return float(0.5 * feat + 0.5 * entropy)


# ------------------ Summarization utilities (kept; safe) ------------------

def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class SummarizationExampleDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def build_summarization_collate(tokenizer, max_input_len: int, max_target_len: int):
    pad_token_id = tokenizer.pad_token_id

    def _collate(batch: List[Dict]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        inputs = tokenizer(
            [ex["U"] for ex in batch],
            padding=True,
            truncation=True,
            max_length=max_input_len,
            return_tensors="pt",
        )
        # (deprecated API in some HF versions; but works)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                [ex["Y"] for ex in batch],
                padding=True,
                truncation=True,
                max_length=max_target_len,
                return_tensors="pt",
            )

        label_ids = labels["input_ids"]
        label_ids[label_ids == pad_token_id] = -100

        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": label_ids,
            "decoder_attention_mask": labels["attention_mask"],
        }
        targets = [ex["Y"] for ex in batch]
        return model_inputs, targets

    return _collate


# ------------------ LoRA / parameter exchange helpers ------------------

def collect_lora_keys(state_dict_keys, include_head: bool = False):
    keys = [k for k in state_dict_keys if "lora_" in k]
    if include_head:
        keys += [k for k in state_dict_keys if k.startswith("classifier.")]
    return sorted(set(keys))


def get_named_tensors(model, keys):
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy() for k in keys]


def set_named_tensors_(model, keys, tensors):
    with torch.no_grad():
        sd = model.state_dict()
        for k, arr in zip(keys, tensors):
            sd[k].copy_(torch.tensor(arr))


# ------------------ Model builders ------------------

def build_vit_adalora(num_classes: int):
    hf_model_name = getattr(cfg, "hf_model_name", "nateraw/vit-base-patch16-224-cifar10")
    lora_r = int(getattr(cfg, "lora_r", 8))
    lora_alpha = int(getattr(cfg, "lora_alpha", 16))
    lora_dropout = float(getattr(cfg, "lora_dropout", 0.1))
    target_modules = getattr(cfg, "class_lora_target_modules", ["query", "key", "value"])

    model = ViTForImageClassification.from_pretrained(
        hf_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    adalora_cfg = AdaLoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["classifier"],
        init_r=lora_r,
        tinit=0,
        tfinal=int(getattr(cfg, "adalora_tfinal", 0)),
        deltaT=int(getattr(cfg, "adalora_deltaT", 1)),
        beta1=float(getattr(cfg, "adalora_beta1", 0.85)),
        beta2=float(getattr(cfg, "adalora_beta2", 0.85)),
        orth_reg_weight=float(getattr(cfg, "adalora_orth_reg", 0.0)),
    )
    model = get_peft_model(model, adalora_cfg)

    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)  # only LoRA params trainable

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[client] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def build_t5_adalora():
    hf_model_name = getattr(cfg, "hf_summarization_model_name", "google-t5/t5-base")
    lora_r = int(getattr(cfg, "lora_r", 8))
    lora_alpha = int(getattr(cfg, "lora_alpha", max(16, 2 * lora_r)))
    lora_dropout = float(getattr(cfg, "lora_dropout", 0.1))
    target_modules = getattr(cfg, "summ_lora_target_modules", ["q", "k", "v", "o", "wi_0", "wi_1", "wi_2"])

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = T5ForConditionalGeneration.from_pretrained(hf_model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    adalora_cfg = AdaLoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None,
        init_r=lora_r,
        tinit=0,
        tfinal=int(getattr(cfg, "adalora_tfinal", 0)),
        deltaT=int(getattr(cfg, "adalora_deltaT", 1)),
        beta1=float(getattr(cfg, "adalora_beta1", 0.85)),
        beta2=float(getattr(cfg, "adalora_beta2", 0.85)),
        orth_reg_weight=float(getattr(cfg, "adalora_orth_reg", 0.0)),
    )
    model = get_peft_model(model, adalora_cfg)

    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

    return model, tokenizer


def build_vit_preprocessor(model):
    size = int(getattr(model.config, "image_size", 224))
    mean = torch.tensor(getattr(model.config, "image_mean", [0.5, 0.5, 0.5])).view(1, 3, 1, 1)
    std = torch.tensor(getattr(model.config, "image_std", [0.5, 0.5, 0.5])).view(1, 3, 1, 1)

    def _prep(x, device):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.to(device, dtype=torch.float32)
        if torch.max(x) > 1.5:
            x = x / 255.0
        try:
            x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
        except TypeError:
            x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)

        mean_dev, std_dev = mean.to(device), std.to(device)
        x = (x - mean_dev) / std_dev
        return {"pixel_values": x}

    return _prep


# ------------------ Flower Client ------------------

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, device: torch.device, task="vision", tokenizer=None, args=None):
        self.model = model
        self.device = device
        self.task = task
        self.tokenizer = tokenizer
        self.client_id = int(args.id)
        self.fold = int(args.fold)
        self.exchange_head = bool(getattr(args, "exchange_head", False))

        self._vit_prep = build_vit_preprocessor(self.model) if self.task == "vision" else None

        self.lora_keys = collect_lora_keys(self.model.state_dict().keys(), include_head=self.exchange_head)

        lora_lr = float(getattr(cfg, "lora_lr", getattr(cfg, "lr", 5e-4)))
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lora_lr,
            weight_decay=float(getattr(cfg, "weight_decay", 0.05)),
        )

        self.accumulation_steps = int(getattr(cfg, "accumulation_steps", 1))
        self.client_eval_ratio = float(getattr(cfg, "client_eval_ratio", 0.2))
        self.n_samples_clients = int(getattr(cfg, "n_samples_clients", -1))

    # -------- data loading --------

    def load_current_data(self, cur_round: int, train=True) -> DataLoader:
        if self.task != "vision":
            raise NotImplementedError("This corrected pipeline focuses on CIFAR10 vision data.")

        path = DATA_ROOT / f"client_{self.client_id}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python tools/make_anda_splits.py --download "
                f"--num_clients {getattr(cfg, 'n_clients', 15)} --out_dir data/cur_datasets"
            )

        cur_data = np.load(path, allow_pickle=True).item()
        X = torch.tensor(cur_data["train_features"], dtype=torch.float32)  # [N,3,32,32] uint8 -> float
        y = torch.tensor(cur_data["train_labels"], dtype=torch.int64)

        # optional subsample
        if self.n_samples_clients > 0:
            X = X[: self.n_samples_clients]
            y = y[: self.n_samples_clients]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.client_eval_ratio, random_state=int(getattr(cfg, "random_seed", 42)) + self.fold
        )

        if train:
            ds = CombinedDataset(X_train, y_train, transform=None)
            return DataLoader(ds, batch_size=int(getattr(cfg, "batch_size", 8)), shuffle=True)
        else:
            ds = CombinedDataset(X_train, y_train, transform=None)
            return DataLoader(ds, batch_size=int(getattr(cfg, "test_batch_size", 16)), shuffle=False)

    # -------- Flower param exchange --------

    def get_parameters(self, config):
        return get_named_tensors(self.model, self.lora_keys)

    def set_parameters(self, parameters):
        set_named_tensors_(self.model, self.lora_keys, parameters)

    def _forward_prep(self, x):
        return self._vit_prep(x, self.device)

    # -------- training --------

    def _train_one_epoch(self, loader):
        self.model.train()
        running = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(loader, start=1):
            model_inputs = self._forward_prep(xb)
            model_inputs["labels"] = yb.to(self.device)

            out = self.model(**model_inputs)
            loss = out.loss
            running += float(loss.item())

            (loss / self.accumulation_steps).backward()

            if step % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        if (len(loader) % self.accumulation_steps) != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return running / max(1, len(loader))

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total, correct = 0, 0
        running_loss = 0.0

        for xb, yb in loader:
            model_inputs = self._forward_prep(xb)
            model_inputs["labels"] = yb.to(self.device)
            out = self.model(**model_inputs)
            loss = out.loss
            logits = out.logits
            preds = logits.argmax(dim=-1)
            total += yb.numel()
            correct += (preds.cpu() == yb).sum().item()
            running_loss += float(loss.item())

        avg_loss = running_loss / max(1, len(loader))
        acc = correct / max(1, total)
        return avg_loss, acc

    # -------- Flower API --------

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        cur_round = int(config.get("current_round", 1))
        local_epochs = int(config.get("local_epochs", getattr(cfg, "local_epochs", 1)))

        train_loader = self.load_current_data(cur_round, train=True)

        # descriptor vector (fixed length)
        desc_vec = vit_descriptor_vector(
            self.model, train_loader, self.device,
            max_batches=int(getattr(cfg, "descriptor_max_batches", 2)),
            k=int(getattr(cfg, "descriptor_k", 64)),
        )
        complexity = compute_complexity_from_vec(desc_vec, k=int(getattr(cfg, "descriptor_k", 64)))

        for _ in range(local_epochs):
            _ = self._train_one_epoch(train_loader)

        torch.cuda.empty_cache()
        gc.collect()

        metrics = {
            "client_id": int(self.client_id),
            "complexity": float(complexity),
            "descriptor": desc_vec.tolist(),  # <-- server clustering needs this
        }
        return self.get_parameters(config), len(train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        cur_round = int(config.get("current_round", 1))
        val_loader = self.load_current_data(cur_round, train=False)
        loss, acc = self._evaluate(val_loader)
        return float(loss), len(val_loader.dataset), {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--exchange_head", action="store_true")
    args = parser.parse_args()

    utils.set_seed(int(getattr(cfg, "random_seed", 42)) + int(args.fold))
    device = utils.check_gpu()
    if isinstance(device, str):
        device = torch.device(device)

    if IS_SUMMARIZATION:
        model, tokenizer = build_t5_adalora()
        model = model.to(device)
        task = "summarization"
    else:
        model = build_vit_adalora(num_classes=int(getattr(cfg, "n_classes", 10))).to(device)
        tokenizer = None
        task = "vision"

    client = FlowerClient(model=model, device=device, task=task, tokenizer=tokenizer, args=args).to_client()
    ip = getattr(cfg, "ip", "127.0.0.1")
    port = int(getattr(cfg, "port", 8080))
    fl.client.start_client(server_address=f"{ip}:{port}", client=client)


if __name__ == "__main__":
    main()
