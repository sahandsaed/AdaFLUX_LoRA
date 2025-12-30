"""
AdaFLUX-LoRA client side (full rewrite based on FedIT structure).

- Uses PEFT AdaLoRA adapters (true AdaLoRA, not just static LoRA).
- Trains only adapter parameters locally and exchanges only adapter weights with the server.
- Adds lightweight descriptor + complexity estimation (AdaFLUX-style) per client/round.

NOTE:
- Server-side clustering / rank scheduling must be handled in the strategy.
- This client only computes descriptors and (optionally) logs/returns a complexity score.
"""

import numpy as np
# Keep Flower compatibility for NumPy>=2.0
if not hasattr(np, "float_"):
    np.float_ = np.float64

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import flwr as fl
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import public.config as cfg
import public.utils as utils
import public.models as models

# --- NEW: LoRA / HF imports ---
from transformers import AutoTokenizer, T5ForConditionalGeneration, ViTForImageClassification
from peft import LoraConfig, AdaLoraConfig, get_peft_model  # LoraConfig kept for compatibility if needed


ROOT_DIR = Path(parent_dir)
DATA_ROOT = ROOT_DIR / "data" / "cur_datasets"
IS_SUMMARIZATION = cfg.dataset_name == "Summarization"


# ============================================================
#   Utilities for LoRA/AdaLoRA rank logging
# ============================================================
def _first_adapter_name(module):
    # PEFT keeps per-adapter dicts like {'default': <Linear>}. Fallback to 'default'.
    try:
        ks = list(module.lora_A.keys())
        return ks[0] if len(ks) > 0 else "default"
    except Exception:
        return "default"


@torch.no_grad()
def compute_lora_delta_ranks(model) -> dict:
    """
    Returns { "<module_path>": {"rank": int, "r_nominal": int, "shape": (out,in)} }
    for each LoRA/AdaLoRA-injected module in the model.
    """
    ranks = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            adapter = _first_adapter_name(mod)

            # A: [r, in], B: [out, r] in PEFT
            A = getattr(mod.lora_A[adapter], "weight", None)
            B = getattr(mod.lora_B[adapter], "weight", None)
            if A is None or B is None:
                continue

            A = A.detach().cpu().float()   # [r, in]
            B = B.detach().cpu().float()   # [out, r]
            deltaW = B @ A                 # [out, in]

            # numeric matrix rank (SVD thresholding handled internally)
            eff_rank = int(torch.linalg.matrix_rank(deltaW).item())

            # nominal r (if available in module.r dict), else A.shape[0]
            try:
                r_nominal = int(getattr(mod, "r", {}).get(adapter, A.shape[0]))
            except Exception:
                r_nominal = int(A.shape[0])

            ranks[name] = {
                "rank": eff_rank,
                "r_nominal": r_nominal,
                "shape": tuple(deltaW.shape),
            }
    return ranks


def append_round_rank_log(base_path: str, client_id: int, round_idx: int, round_dict: dict):
    os.makedirs(base_path, exist_ok=True)
    path = os.path.join(base_path, f"rank_logs_client_{client_id}.npy")
    if os.path.exists(path):
        log = np.load(path, allow_pickle=True).item()
    else:
        log = {}
    log[int(round_idx)] = round_dict
    np.save(path, log)


# ============================================================
#   AdaFLUX-style descriptor & complexity utilities
# ============================================================
def compute_descriptor_from_vit(model, loader, device, max_batches: int = 2) -> Dict:
    """
    Compute a simple descriptor for vision data based on ViT features:
    - mean and variance of pooled patch embeddings
    - label histogram (for up to 32 classes)
    """
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for b_idx, (xb, yb) in enumerate(loader):
            if b_idx >= max_batches:
                break
            xb = xb.to(device, dtype=torch.float32)
            # Minimal preprocessing: use same path as _forward_prep but directly to vit
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)
            if xb.size(1) == 1:
                xb = xb.repeat(1, 3, 1, 1)
            if torch.max(xb) > 1.5:
                xb = xb / 255.0

            size = int(getattr(model.config, "image_size", 224))
            try:
                xb = F.interpolate(xb, size=(size, size), mode="bicubic",
                                   align_corners=False, antialias=True)
            except TypeError:
                xb = F.interpolate(xb, size=(size, size), mode="bicubic",
                                   align_corners=False)

            # model.vit forward
            feats = model.vit(pixel_values=xb).last_hidden_state  # [B, N, D]
            pooled = feats.mean(dim=1)  # [B, D]
            embeddings.append(pooled.cpu())
            labels.append(yb.cpu())

    if not embeddings:
        return {}

    Z = torch.cat(embeddings, dim=0)  # [N, D]
    Y = torch.cat(labels, dim=0)      # [N]

    mu = Z.mean(dim=0)                # [D]
    sigma2 = Z.var(dim=0)             # [D]

    # label histogram with up to 32 classes
    num_classes = min(32, int(Y.max().item()) + 1)
    hist = torch.bincount(Y, minlength=num_classes).float()
    hist = hist / hist.sum().clamp_min(1.0)

    desc = {
        "mu": mu.numpy(),
        "sigma2": sigma2.numpy(),
        "label_hist": hist.numpy().tolist(),
    }
    return desc


def compute_descriptor_from_t5(model, loader, device, max_batches: int = 2) -> Dict:
    """
    Compute a simple descriptor for summarization data based on encoder features:
    - mean and variance of pooled encoder hidden states.
    We ignore labels/targets for entropy in this example.
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for b_idx, (model_inputs, _) in enumerate(loader):
            if b_idx >= max_batches:
                break
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)

            encoder_outputs = model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            feats = encoder_outputs.last_hidden_state  # [B, T, D]
            pooled = feats.mean(dim=1)  # [B, D]
            embeddings.append(pooled.cpu())

    if not embeddings:
        return {}

    Z = torch.cat(embeddings, dim=0)  # [N, D]
    mu = Z.mean(dim=0)                # [D]
    sigma2 = Z.var(dim=0)             # [D]

    desc = {
        "mu": mu.numpy(),
        "sigma2": sigma2.numpy(),
        # no label_hist here, but we keep interface consistent
    }
    return desc


def compute_complexity_score_from_descriptor(desc: Dict) -> float:
    """
    AdaFLUX-like scalar complexity score:
      - feature variance norm
      - label entropy (if available)
    """
    if not desc:
        return 0.0

    sigma2 = np.asarray(desc["sigma2"])
    feat_score = float(np.linalg.norm(sigma2))

    entropy_score = 0.0
    if "label_hist" in desc:
        p = np.asarray(desc["label_hist"], dtype=np.float64) + 1e-9
        p = p / p.sum()
        entropy_score = float(-(p * np.log(p)).sum())

    complexity = 0.5 * feat_score + 0.5 * entropy_score
    return float(complexity)


# ============================================================
#   Summarization helpers
# ============================================================
def summarization_scenario_from_non_iid(non_iid: str) -> str:
    if IS_SUMMARIZATION:
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
            # same
            "PX": "PX",
            "PY": "PY",
            "PY_given_X": "PY_given_X",
            "PX_given_Y": "PX_given_Y",
        }
        if non_iid not in mapping:
            allowed = ", ".join(sorted(mapping.keys()))
            raise ValueError(f"Unknown non_iid '{non_iid}'. Allowed values: {allowed}")
        return mapping[non_iid]
    else:
        return non_iid


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


def _lcs_length(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> int:
    a = list(a_tokens)
    b = list(b_tokens)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_tokens = pred.strip().split()
    ref_tokens = ref.strip().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    prec = lcs / max(1, len(pred_tokens))
    rec = lcs / max(1, len(ref_tokens))
    denom = prec + rec
    return 0.0 if denom == 0 else 2 * prec * rec / denom


# ============================================================
#   Utilities for FedIT-style sync (still used)
# ============================================================
def collect_lora_keys(state_dict_keys, include_head=False):
    """Return a stable, sorted list of LoRA/AdaLoRA (and optional head) parameter keys."""
    lora_keys = [k for k in state_dict_keys if "lora_" in k]
    if include_head:
        # typical classifier keys in ViTForImageClassification
        lora_keys += [k for k in state_dict_keys if k.startswith("classifier.")]
    lora_keys = sorted(set(lora_keys))
    return lora_keys


def get_named_tensors(model, keys):
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy() for k in keys]


def set_named_tensors_(model, keys, tensors):
    """In-place update of a subset of state_dict tensors by key order."""
    with torch.no_grad():
        sd = model.state_dict()
        for k, arr in zip(keys, tensors):
            sd[k].copy_(torch.tensor(arr))


# ============================================================
#   Build HF ViT + PEFT AdaLoRA
# ============================================================
def build_vit_adalora(num_classes: int):
    # Defaults (override in cfg if present)
    hf_model_name = getattr(cfg, "hf_model_name", "google/vit-base-patch16-224-in21k")
    lora_r = int(getattr(cfg, "lora_r", 32))           # max rank
    lora_alpha = int(getattr(cfg, "lora_alpha", 64))
    lora_dropout = float(getattr(cfg, "lora_dropout", 0.00))
    target_modules = getattr(cfg, "class_lora_target_modules", ["query", "key", "value"])

    # Base model (classifier is created with correct num_labels)
    model = ViTForImageClassification.from_pretrained(
        hf_model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # True AdaLoRA (adaptive low-rank adaptation)
    adalora_cfg = AdaLoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["classifier"],
        init_r=lora_r,
        tinit=0,
        tfinal=int(getattr(cfg, "adalora_tfinal", 0)),  # 0 uses default schedule
        deltaT=int(getattr(cfg, "adalora_deltaT", 1)),
        beta1=float(getattr(cfg, "adalora_beta1", 0.85)),
        beta2=float(getattr(cfg, "adalora_beta2", 0.85)),
        orth_reg_weight=float(getattr(cfg, "adalora_orth_reg", 0.0)),
    )

    model = get_peft_model(model, adalora_cfg)

    # Freeze everything except LoRA/AdaLoRA params
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[AdaFLUX-LoRA] Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%) across {len(trainable_names)} tensors.")

    return model


def build_t5_adalora():
    hf_model_name = getattr(cfg, "hf_summarization_model_name", "google-t5/t5-base")
    lora_r = int(getattr(cfg, "lora_r", 8))
    lora_alpha = int(getattr(cfg, "lora_alpha", max(16, 2 * lora_r)))
    lora_dropout = float(getattr(cfg, "lora_dropout", 0.1))
    target_modules = getattr(
        cfg,
        "summ_lora_target_modules",
        ["q", "k", "v", "o", "wi_0", "wi_1", "wi_2"],
    )

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

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[AdaFLUX-LoRA] T5 trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    return model, tokenizer


def build_vit_preprocessor(model):
    size = int(getattr(model.config, "image_size", 224))
    mean = torch.tensor(getattr(model.config, "image_mean", [0.5, 0.5, 0.5])).view(1, 3, 1, 1)
    std = torch.tensor(getattr(model.config, "image_std", [0.5, 0.5, 0.5])).view(1, 3, 1, 1)

    def _prep(x, device):
        # x: [B,H,W] or [B,1,H,W] or [B,3,H,W], dtype float32 (0–1 or 0–255)
        if x.dim() == 3:
            x = x.unsqueeze(1)                # -> [B,1,H,W]
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)          # -> [B,3,H,W]
        x = x.to(device, dtype=torch.float32)

        # If likely in 0–255, scale to 0–1
        if torch.max(x) > 1.5:
            x = x / 255.0

        # Resize to ViT expected size
        try:
            x = F.interpolate(x, size=(size, size), mode="bicubic",
                               align_corners=False, antialias=True)
        except TypeError:
            x = F.interpolate(x, size=(size, size), mode="bicubic",
                               align_corners=False)

        # Normalize
        mean_dev, std_dev = mean.to(device), std.to(device)
        x = (x - mean_dev) / std_dev
        return {"pixel_values": x}

    return _prep


# ============================================================
#   Flower client class
# ============================================================
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        device: torch.device,
        task: str = "vision",
        tokenizer=None,
        args=None,
    ):
        self.model = model
        self.client_id = args.id  # [0, cfg.n_clients]
        self.device = device
        self.exchange_head = args.exchange_head
        self.accumulation_steps = int(getattr(cfg, "accumulation_steps", 1))
        self.task = task
        self.tokenizer = tokenizer
        self.fold = args.fold
        self.args = args

        if self.task == "vision":
            self._vit_prep = build_vit_preprocessor(self.model)
            self._summ_collate = None
        else:
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be provided for summarization task.")
            self._vit_prep = None
            self.max_input_len = int(getattr(cfg, "summ_max_input_length", 512))
            self.max_target_len = int(getattr(cfg, "summ_max_target_length", 128))
            self.gen_max_target_len = int(
                getattr(cfg, "summ_gen_max_target_length", self.max_target_len)
            )
            self.eval_num_beams = int(getattr(cfg, "summ_eval_num_beams", 4))
            self._summ_collate = build_summarization_collate(
                self.tokenizer, self.max_input_len, self.max_target_len
            )
            self._init_summarization_partitions()

        # Determine which params we share (only LoRA/AdaLoRA by default)
        self.lora_keys = collect_lora_keys(
            self.model.state_dict().keys(), include_head=self.exchange_head
        )

        self.drifting_log = []
        self.metrics = {"rounds": [], "loss": [], "accuracy": [], "complexity": []}

        if cfg.training_drifting and self.task == "vision":
            drifting_path = DATA_ROOT / "drifting_log.npy"
            if drifting_path.exists():
                drifting_log = np.load(drifting_path, allow_pickle=True).item()
                self.drifting_log = drifting_log[self.client_id]

        # Optimizer on trainable (AdaLoRA) only
        lora_lr = float(getattr(cfg, "lora_lr", getattr(cfg, "lr", 5e-3)))
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lora_lr,
            weight_decay=float(getattr(cfg, "weight_decay", 0.05)),
        )

    # ----------------------------- summarization partitioning -----------------------------
    def _init_summarization_partitions(self) -> None:
        scenario = summarization_scenario_from_non_iid(self.args.non_iid_type)
        base_dir = DATA_ROOT / scenario
        seed_dir = base_dir / f"seed_{self.fold + 1}"
        if not seed_dir.exists():
            seed_dir = base_dir / "seed_1"
        if not seed_dir.exists():
            raise FileNotFoundError(
                f"Could not locate summarization data directory for scenario '{scenario}'. Expected at {seed_dir}."
            )
        self.summarization_dir = seed_dir

        train_path = seed_dir / f"client_{self.client_id}_train.jsonl"
        test_path = seed_dir / f"client_{self.client_id}_test.jsonl"

        self._summ_train_samples = read_jsonl(train_path)
        self._summ_eval_samples = read_jsonl(test_path) if test_path.exists() else []

        max_local_samples = int(getattr(cfg, "n_samples_clients", -1))
        if max_local_samples > 0 and len(self._summ_train_samples) > max_local_samples:
            rng = np.random.default_rng(cfg.random_seed + self.client_id)
            subset_idx = np.sort(
                rng.choice(
                    len(self._summ_train_samples), size=max_local_samples, replace=False
                )
            )
            self._summ_train_samples = [self._summ_train_samples[i] for i in subset_idx]

    def _summarization_loader(self, train: bool) -> DataLoader:
        samples = self._summ_train_samples if train else self._summ_eval_samples
        dataset = SummarizationExampleDataset(samples)
        batch_size = cfg.batch_size if train else cfg.test_batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train and len(dataset) > 0,
            collate_fn=self._summ_collate,
        )

    # ----------------------------- data loading -----------------------------
    def load_current_data(self, cur_round, train=True) -> DataLoader:
        if self.task == "summarization":
            return self._summarization_loader(train=train)

        # Vision
        if not cfg.training_drifting:
            cur_data = np.load(
                f"../data/cur_datasets/client_{self.client_id}.npy", allow_pickle=True
            ).item()
        else:
            load_index = max([idx for idx in self.drifting_log if idx <= cur_round], default=0)
            cur_data = np.load(
                f"../data/cur_datasets/client_{self.client_id}_round_{load_index}.npy",
                allow_pickle=True,
            ).item()

        cur_features = (
            torch.tensor(cur_data["train_features"], dtype=torch.float32)
            if not cfg.training_drifting
            else torch.tensor(cur_data["features"], dtype=torch.float32)
        )
        cur_labels = (
            torch.tensor(cur_data["train_labels"], dtype=torch.int64)
            if not cfg.training_drifting
            else torch.tensor(cur_data["labels"], dtype=torch.int64)
        )

        cur_features = cur_features.unsqueeze(1) if utils.get_in_channels() == 1 else cur_features

        # Split train/val
        train_features, val_features, train_labels, val_labels = train_test_split(
            cur_features, cur_labels, test_size=cfg.client_eval_ratio, random_state=cfg.random_seed
        )

        # Optionally reduce local samples
        if cfg.n_samples_clients > 0:
            train_features = train_features[: cfg.n_samples_clients]
            train_labels = train_labels[: cfg.n_samples_clients]

        if train:
            ds = models.CombinedDataset(train_features, train_labels, transform=None)
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
        else:
            ds = models.CombinedDataset(val_features, val_labels, transform=None)
            return DataLoader(ds, batch_size=cfg.test_batch_size, shuffle=False)

    # ----------------------------- Flower interface: params -----------------------------
    def get_parameters(self, config):
        return get_named_tensors(self.model, self.lora_keys)

    def set_parameters(self, parameters):
        set_named_tensors_(self.model, self.lora_keys, parameters)

    def _forward_prep(self, x):
        if self.task != "vision":
            raise RuntimeError("Forward prep is only defined for vision inputs.")
        return self._vit_prep(x, self.device)

    # ----------------------------- training loops -----------------------------
    def _train_one_epoch(self, loader):
        self.model.train()
        running = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=1):
            if self.task == "vision":
                xb, yb = batch
                model_inputs = self._forward_prep(xb)
                model_inputs["labels"] = yb.to(self.device)
            else:
                model_inputs, _ = batch
                model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

            out = self.model(**model_inputs)
            loss = out.loss
            running += float(loss.item())

            (loss / self.accumulation_steps).backward()

            if step % self.accumulation_steps == 0:
                # Optional gradient clipping here
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        # flush leftover micro-batches if the epoch ended mid-accumulation
        if (len(loader) % self.accumulation_steps) != 0:
            # Optional gradient clipping here
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return running / max(1, len(loader))

    def _train_for_iterations(self, loader, num_iterations: int):
        self.model.train()
        running = 0.0
        data_iter = iter(loader)
        self.optimizer.zero_grad(set_to_none=True)

        for i in range(1, num_iterations + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            if self.task == "vision":
                xb, yb = batch
                model_inputs = self._forward_prep(xb)
                model_inputs["labels"] = yb.to(self.device)
            else:
                model_inputs, _ = batch
                model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

            out = self.model(**model_inputs)
            loss = out.loss
            running += float(loss.item())

            (loss / self.accumulation_steps).backward()

            if i % self.accumulation_steps == 0:
                # Optional gradient clipping
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        # flush leftover micro-batches
        if (num_iterations % self.accumulation_steps) != 0:
            # Optional gradient clipping
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        return running / max(1, num_iterations)

    # ----------------------------- evaluation -----------------------------
    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0

        if self.task == "vision":
            total, correct = 0, 0
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

        # Summarization evaluation (ROUGE-L F1 as accuracy proxy)
        metric_total = 0.0
        example_count = 0
        for model_inputs, _ in loader:
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            outputs = self.model(**model_inputs)
            loss = outputs.loss
            running_loss += float(loss.item())

            generated_ids = self.model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=self.gen_max_target_len,
                num_beams=self.eval_num_beams,
            )
            preds = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            labels = model_inputs["labels"].detach().cpu().clone()
            labels[labels == -100] = self.tokenizer.pad_token_id
            targets = self.tokenizer.batch_decode(
                labels,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            for pred, ref in zip(preds, targets):
                metric_total += rouge_l_f1(pred, ref)
                example_count += 1

        avg_loss = running_loss / max(1, len(loader))
        avg_metric = metric_total / max(1, example_count)
        return avg_loss, avg_metric

    # ----------------------------- fit (AdaFLUX-LoRA) -----------------------------
    def fit(self, parameters, config):
        # Receive global adapter weights
        self.set_parameters(parameters)

        cur_round = config["current_round"]
        train_loader = self.load_current_data(cur_round, train=True)
        print(
            f"[AdaFLUX-LoRA] Client {self.client_id} - Round {cur_round} - "
            f"Training samples: {len(train_loader.dataset)}"
        )

        # ---------- Compute AdaFLUX descriptor + complexity score ----------
        try:
            if self.task == "vision":
                desc = compute_descriptor_from_vit(
                    self.model, train_loader, self.device,
                    max_batches=int(getattr(cfg, "descriptor_max_batches", 2)),
                )
            else:
                desc = compute_descriptor_from_t5(
                    self.model, train_loader, self.device,
                    max_batches=int(getattr(cfg, "descriptor_max_batches", 2)),
                )
            complexity = compute_complexity_score_from_descriptor(desc)
        except Exception as e:
            print(f"[AdaFLUX-LoRA] Descriptor computation failed on Client {self.client_id}: {e}")
            desc = {}
            complexity = 0.0

        print(f"[AdaFLUX-LoRA] Client {self.client_id} - Round {cur_round} - Complexity score: {complexity:.4f}")

        # ---------- Local training ----------
        raw_iterations = config.get("local_iterations", 0)
        local_iterations = int(raw_iterations) if raw_iterations else 0
        if local_iterations > 0:
            print(
                f"[AdaFLUX-LoRA] Client {self.client_id} - Round {cur_round} - "
                f"Training for {local_iterations} iterations"
            )
            _ = self._train_for_iterations(train_loader, local_iterations)
        else:
            local_epochs = int(config.get("local_epochs", 1))
            print(
                f"[AdaFLUX-LoRA] Client {self.client_id} - Round {cur_round} - "
                f"epochs={local_epochs} | lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )
            for _ in range(local_epochs):
                _ = self._train_one_epoch(train_loader)

        # free cuda memory
        torch.cuda.empty_cache()
        gc.collect()

        # rank logging
        if cfg.calculate_ranks:
            rank_dict = compute_lora_delta_ranks(self.model)
            append_round_rank_log(
                base_path=f"histories/rank_logs/{cfg.default_path}",
                client_id=self.client_id,
                round_idx=cur_round,
                round_dict=rank_dict,
            )
            print(
                f"[AdaFLUX-LoRA] Client {self.client_id} - Round {cur_round} - LoRA ranks: "
                + ", ".join(
                    [f"{k}: {v['rank']}/{v['r_nominal']}" for k, v in rank_dict.items()]
                )
            )

        # Return updated adapter weights + meta metrics
        metrics = {
            "complexity": float(complexity),
            "client_id": int(self.client_id),
        }
        return self.get_parameters(config), len(train_loader.dataset), metrics

    # ----------------------------- evaluate -----------------------------
    def evaluate(self, parameters, config):
        # Use aggregated adapter params for evaluation
        self.set_parameters(parameters)

        cur_round = config["current_round"]
        val_loader = self.load_current_data(cur_round, train=False)
        print(
            f"[AdaFLUX-LoRA] Client {self.client_id} - Round {cur_round} - "
            f"Evaluating on {len(val_loader.dataset)} samples"
        )

        loss, acc = self._evaluate(val_loader)

        # Log & save for plots
        metric_label = "Accuracy" if self.task == "vision" else "ROUGE-L"
        print(
            f"[AdaFLUX-LoRA] Client {self.client_id} - Round {cur_round} - "
            f"Loss: {loss:.4f}, {metric_label}: {acc:.4f}"
        )
        self.metrics["rounds"].append(cur_round)
        self.metrics["loss"].append(loss)
        self.metrics["accuracy"].append(acc)
        # complexity of this round (if any) is appended in fit only
        os.makedirs(f"results/{cfg.default_path}", exist_ok=True)
        np.save(f"results/{cfg.default_path}/client_{self.client_id}_metrics.npy", self.metrics)

        # free cuda memory
        torch.cuda.empty_cache()
        gc.collect()

        # Flower expects (loss, num_examples, metrics_dict)
        return float(loss), len(val_loader.dataset), {"accuracy": float(acc)}


# ============================================================
#   main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Flower + AdaLoRA (AdaFLUX-LoRA client)")
    parser.add_argument(
        "--id",
        type=int,
        choices=range(0, cfg.n_clients),
        required=True,
        help="Specifies the artificial data partition",
    )
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold")
    # Optional: exchange head too (default False)
    parser.add_argument(
        "--exchange_head",
        action="store_true",
        help="Also aggregate classifier head",
    )
    parser.add_argument(
        "--non_iid_type",
        type=str,
        default=cfg.non_iid_type,
        help="Type of non-IID data partitioning",
    )
    args = parser.parse_args()

    args.exchange_head = False  # keep consistent with FedIT unless explicitly changed

    # Seed & device
    utils.set_seed(cfg.random_seed + args.fold)
    device = utils.check_gpu(client_id=args.id)

    if cfg.dataset_name == "Summarization":
        model, tokenizer = build_t5_adalora()
        model = model.to(device)
        task = "summarization"
    else:
        model = build_vit_adalora(num_classes=cfg.n_classes).to(device)
        tokenizer = None
        task = "vision"

    # Start Flower client
    client = FlowerClient(
        model=model,
        device=device,
        task=task,
        tokenizer=tokenizer,
        args=args,
    ).to_client()
    fl.client.start_client(server_address=f"{cfg.ip}:{cfg.port}", client=client)


if __name__ == "__main__":
    main()
