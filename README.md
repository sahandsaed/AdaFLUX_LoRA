# AdaFLUX-LoRA: Federated Adaptive LoRA with FLUX-style Clustering

AdaFLUX-LoRA is a federated fine-tuning framework that combines:

- **FLUX-style client descriptors** (data + gradient statistics)
- **Dynamic clustering** of clients based on descriptors
- **Cluster-wise LoRA aggregation** (CFL-style federation)
- **AdaLoRA-style rank adaptation** (handled inside LoRA modules)
- **Routing for unseen clients** based on learned clusters

The codebase is built on top of **Flower**, **PEFT**, and **Transformers**, and currently supports:
- Vision (ViT) and summarization (T5) tasks
- Both **online FL** (server + clients) and **sequential local simulation**

---

## 📁 Repository Structure

```text
.
├── client_adaflux_lora.py       # Flower client (AdaFLUX-LoRA)
├── server_adaflux_lora.py       # Flower server (AdaFLUX-LoRA)
├── fedseq_adaflux_lora.py       # Sequential local FL simulation (no networking)
├── logging_utils.py             # TensorBoard logger helper
├── router.py                    # FLUXRouter for test-time routing
├── visualize_clusters.py        # Cluster embedding visualization (e.g., PCA/UMAP)
├── public/
│   ├── config.py                # Global config (dataset, model, FL hyperparameters)
│   ├── utils.py                 # Plotting, seeding, GPU utils, folder creation
│   ├── models.py                # CombinedDataset, ViT/T5 helpers if needed
├── data/
│   └── cur_datasets/            # Pre-partitioned data per client (vision or summarization)
├── checkpoints/                 # Server-side LoRA checkpoints (online FL)
├── checkpoints_local/           # Local-simulation LoRA checkpoints
├── results/                     # Evaluation metrics (online FL)
├── results_local/               # Metrics from sequential simulation
├── requirements.txt
└── README.md
