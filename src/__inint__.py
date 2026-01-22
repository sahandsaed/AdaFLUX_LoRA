"""
AdaFLUX-LoRA Federated Learning Framework

This package provides:
 - Adaptive cluster-based routing (FLUX)
 - Personalized LoRA rank adaptation (AdaLoRA)
 - Federated simulation + client/server training
 - Cluster visualization + diagnostic tools
"""

from .client import FlowerClient
from .router import FLUXRouter
from .server_adaflux_lora_cluster import AdaFLUXLoRAServer  # keep if exists; else remove
from .visualize_clusters import plot_flux_embeddings

__all__ = [
    "FlowerClient",
    "FLUXRouter",
    "AdaFLUXLoRAServer",
    "plot_flux_embeddings",
]
