import os, sys, json
import numpy as np
import torch
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Point to repo root
PROJECT_ROOT = Path("/content/AdaFLUX_LoRA").resolve()
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import config as cfg
from src.client import build_vit_adalora, set_named_tensors_, build_vit_preprocessor

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CLUSTER_META_PATH = PROJECT_ROOT / "results" / "cluster_assignments.json"

if not CLUSTER_META_PATH.exists():
    raise FileNotFoundError(f"Missing {CLUSTER_META_PATH}. Re-run fedseq with metadata saving enabled.")

with open(CLUSTER_META_PATH, "r") as f:
    cluster_info = json.load(f)

cluster_assignments = cluster_info["client_to_cluster"]
print("Loaded cluster map:", cluster_assignments)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_base_model():
    return build_vit_adalora(num_classes=int(cfg.n_classes))

# ---- restore client models ----
client_models = {}
for cid, ckpt in cluster_info["client_checkpoints"].items():
    cid = str(cid)
    model = load_base_model().to(device)
    state = torch.load(ckpt, map_location=device)

    keys = state["keys"]
    tensors = [t.detach().cpu().numpy() for t in state["tensors"]]
    set_named_tensors_(model, keys, tensors)

    model.eval()
    client_models[cid] = model

# ---- restore global model ----
global_model = load_base_model().to(device)
state_global = torch.load(cluster_info["global_checkpoint"], map_location=device)
keys_g = state_global["keys"]
tensors_g = [t.detach().cpu().numpy() for t in state_global["tensors"]]
set_named_tensors_(global_model, keys_g, tensors_g)
global_model.eval()

print("Loaded:", len(client_models), "client models + global model")

# ---- probe loader ----
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

test_ds = CIFAR10(
    root=str(PROJECT_ROOT/"data"),
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
probe_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# Use same preprocessing pipeline as training
prep = build_vit_preprocessor(global_model)

def forward_logits(model, xb):
    inputs = prep(xb, device)
    out = model(**inputs)
    return out.logits

@torch.no_grad()
def accuracy(model, loader, max_batches=None):
    correct, total = 0, 0
    for bi, (xb, yb) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        xb = xb.to(device); yb = yb.to(device)
        logits = forward_logits(model, xb)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)

# Accuracy (optionally limit batches for speed)
acc_global = accuracy(global_model, probe_loader, max_batches=50)
acc_clients = {cid: accuracy(m, probe_loader, max_batches=50) for cid, m in client_models.items()}
print("Global acc (subset):", acc_global)
print("Client acc sample:", list(acc_clients.items())[:5])

import torch.nn.functional as F

@torch.no_grad()
def js_divergence_pair(model_a, model_b, loader, max_batches=20):
    vals = []
    for bi, (xb, yb) in enumerate(loader):
        if bi >= max_batches:
            break
        xb = xb.to(device)
        Pa = F.softmax(forward_logits(model_a, xb), dim=-1)
        Pb = F.softmax(forward_logits(model_b, xb), dim=-1)
        M = 0.5 * (Pa + Pb)
        js = 0.5 * (
            torch.sum(Pa * (torch.log(Pa+1e-9) - torch.log(M+1e-9)), dim=-1) +
            torch.sum(Pb * (torch.log(Pb+1e-9) - torch.log(M+1e-9)), dim=-1)
        )
        vals.append(js.mean().item())
    return float(np.mean(vals))

@torch.no_grad()
def js_divergence_matrix(models: dict, loader, max_batches=20):
    ids = list(models.keys())
    C = len(ids)
    mat = np.zeros((C, C), dtype=np.float32)
    for i in range(C):
        for j in range(i+1, C):
            mat[i, j] = mat[j, i] = js_divergence_pair(models[ids[i]], models[ids[j]], loader, max_batches=max_batches)
    return ids, mat

ids, full_js = js_divergence_matrix(client_models, probe_loader, max_batches=20)
global_compare = [js_divergence_pair(client_models[cid], global_model, probe_loader, max_batches=20) for cid in ids]

pairs_within, pairs_across = [], []
for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        ci, cj = str(ids[i]), str(ids[j])
        js = js_divergence_pair(client_models[ci], client_models[cj], probe_loader, max_batches=20)
        if str(cluster_assignments[ci]) == str(cluster_assignments[cj]):
            pairs_within.append(js)
        else:
            pairs_across.append(js)

print("Mean JS within cluster:", float(np.mean(pairs_within)) if pairs_within else None)
print("Mean JS across cluster:", float(np.mean(pairs_across)) if pairs_across else None)
print("JS vs global:", dict(zip(ids, global_compare)))

sns.set(font_scale=1.0)
plt.figure(figsize=(10, 8))
sns.heatmap(full_js, annot=False, xticklabels=ids, yticklabels=ids, cmap="viridis")
plt.title("JS Divergence Between Client Models (subset)")
plt.show()

plt.figure(figsize=(10, 4))
sns.barplot(x=ids, y=global_compare)
plt.ylabel("JS Divergence to Global (subset)")
plt.title("Client divergence from global model")
plt.xticks(rotation=90)
plt.show()
