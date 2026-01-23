import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Make repo root importable
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import config as cfg
import utils

from transformers import ViTForImageClassification
from peft import AdaLoraConfig, get_peft_model

DATA_ROOT = ROOT_DIR / cfg.out_dir


class CombinedDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


@torch.no_grad()
def vit_descriptor_vector(model, loader, device, max_batches: int = 2, k: int = 64) -> np.ndarray:
    """
    Fixed-length descriptor:
      concat(mean[:k], var[:k], label_hist[:10]) => length 2k + 10
    """
    model.eval()
    embs, labels = [], []

    for b_idx, (xb, yb) in enumerate(loader):
        if b_idx >= max_batches:
            break
        xb = xb.to(device, dtype=torch.float32)

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

        feats = model.vit(pixel_values=xb).last_hidden_state
        pooled = feats.mean(dim=1)
        embs.append(pooled.detach().cpu())
        labels.append(yb.detach().cpu())

    if not embs:
        return np.zeros((2 * k + 10,), dtype=np.float32)

    Z = torch.cat(embs, dim=0)
    Y = torch.cat(labels, dim=0).to(torch.int64)

    mu = Z.mean(dim=0)[:k]
    var = Z.var(dim=0)[:k]

    hist = torch.bincount(Y, minlength=10).float()
    hist = hist / hist.sum().clamp_min(1.0)

    vec = torch.cat([mu, var, hist], dim=0).numpy().astype(np.float32)
    return vec


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


def build_vit_adalora(num_classes: int):
    """
    Real AdaLoRA:
    - uses AdaLoRA schedule parameters (tinit/tfinal/deltaT, betas, orth reg)
    - trains LoRA + classifier head (recommended for CIFAR10)
    """
    model = ViTForImageClassification.from_pretrained(
        cfg.hf_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    adalora_cfg = AdaLoraConfig(
        r=cfg.lora_r,
        init_r=cfg.lora_r,
        tinit=cfg.adalora_tinit,
        tfinal=cfg.adalora_tfinal,
        deltaT=cfg.adalora_deltaT,
        beta1=cfg.adalora_beta1,
        beta2=cfg.adalora_beta2,
        orth_reg_weight=cfg.adalora_orth_reg,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.class_lora_target_modules,
        bias="none",
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, adalora_cfg)

    # Train LoRA params + classifier head
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n) or n.startswith("classifier.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[client] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


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


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, device: torch.device, args=None):
        self.model = model
        self.device = device
        self.client_id = int(args.id)
        self.fold = int(args.fold)
        self.exchange_head = bool(getattr(args, "exchange_head", False))

        self._vit_prep = build_vit_preprocessor(self.model)
        self.lora_keys = collect_lora_keys(self.model.state_dict().keys(), include_head=self.exchange_head)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(getattr(cfg, "lr", 5e-4)),
            weight_decay=float(getattr(cfg, "weight_decay", 0.05)),
        )

        self.accumulation_steps = int(getattr(cfg, "accumulation_steps", 1))

    def load_current_data(self, train=True) -> DataLoader:
        path = DATA_ROOT / f"client_{self.client_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run tools/make_split.py first.")

        cur_data = np.load(path, allow_pickle=True).item()
        X = torch.tensor(cur_data["train_features"], dtype=torch.float32)  # uint8->float
        y = torch.tensor(cur_data["train_labels"], dtype=torch.int64)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=float(getattr(cfg, "client_eval_ratio", 0.2)),
            random_state=int(cfg.random_seed) + self.fold
        )

        if train:
            ds = CombinedDataset(X_train, y_train, transform=None)
            return DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True)
        else:
            ds = CombinedDataset(X_val, y_val, transform=None)
            return DataLoader(ds, batch_size=int(cfg.test_batch_size), shuffle=False)

    def get_parameters(self, config):
        return get_named_tensors(self.model, self.lora_keys)

    def set_parameters(self, parameters):
        set_named_tensors_(self.model, self.lora_keys, parameters)

    def _forward_prep(self, x):
        return self._vit_prep(x, self.device)

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

        return running_loss / max(1, len(loader)), correct / max(1, total)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", cfg.local_epochs))

        train_loader = self.load_current_data(train=True)

        # descriptor (optional for server clustering)
        desc_vec = vit_descriptor_vector(
            self.model, train_loader, self.device,
            max_batches=int(getattr(cfg, "descriptor_max_batches", 2)),
            k=int(getattr(cfg, "descriptor_k", 64)),
        )

        for _ in range(local_epochs):
            _ = self._train_one_epoch(train_loader)

        torch.cuda.empty_cache()
        gc.collect()

        metrics = {"descriptor": desc_vec.tolist()}
        return self.get_parameters(config), len(train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loader = self.load_current_data(train=False)
        loss, acc = self._evaluate(val_loader)
        return float(loss), len(val_loader.dataset), {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--exchange_head", action="store_true")
    args = parser.parse_args()

    utils.set_seed(int(cfg.random_seed) + int(args.fold))
    device = utils.check_gpu()

    model = build_vit_adalora(num_classes=int(cfg.n_classes)).to(device)

    client = FlowerClient(model=model, device=device, args=args).to_client()
    fl.client.start_client(server_address=f"{cfg.ip}:{cfg.port}", client=client)


if __name__ == "__main__":
    main()
