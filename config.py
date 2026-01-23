from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

experiment_name = _cfg.get("experiment_name", "AdaFLUX-LoRA")
seed = _cfg.get("seed", 42)

dataset = _cfg.get("dataset", {})
dataset_name = dataset.get("name", "CIFAR10")
n_classes = int(dataset.get("num_classes", 10))
n_clients = int(dataset.get("num_clients", 15))
non_iid_type = dataset.get("non_iid_type", "label_skew")
non_iid_level = dataset.get("non_iid_level", "medium")
out_dir = dataset.get("out_dir", "data/cur_datasets")

federated = _cfg.get("federated", {})
n_rounds = int(federated.get("rounds", 10))
fit_clients_per_round = int(federated.get("fit_clients_per_round", 3))
eval_clients_per_round = int(federated.get("eval_clients_per_round", 6))
ip = federated.get("ip", "127.0.0.1")
port = int(federated.get("port", 8080))

model = _cfg.get("model", {})
hf_model_name = model.get("base_model", "nateraw/vit-base-patch16-224-cifar10")
lora_r = int(model.get("lora_r", 8))
lora_alpha = int(model.get("lora_alpha", 16))
lora_dropout = float(model.get("lora_dropout", 0.1))
class_lora_target_modules = model.get("lora_target_modules", ["query", "key", "value"])

training = _cfg.get("training", {})
batch_size = int(training.get("batch_size", 8))
test_batch_size = int(training.get("test_batch_size", max(16, 2 * batch_size)))
local_epochs = int(training.get("local_epochs", 1))
lr = float(training.get("lr", 5e-4))
weight_decay = float(training.get("weight_decay", 0.05))

# extra defaults used by client
client_eval_ratio = float(_cfg.get("client_eval_ratio", 0.2))
random_seed = int(_cfg.get("random_seed", seed))

descriptor_max_batches = int(_cfg.get("descriptor_max_batches", 2))
descriptor_k = int(_cfg.get("descriptor_k", 64))

cluster_update_interval = int(_cfg.get("cluster_update_interval", 3))
