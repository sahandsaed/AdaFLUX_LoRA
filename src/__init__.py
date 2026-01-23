"""
AdaFLUX-LoRA package (src)

Exposes:
- FlowerClient
- FLUXRouter (optional)
- plot_flux_embeddings
"""

from .client import FlowerClient
from .router import FLUXRouter
from .visualize_clusters import plot_flux_embeddings

__all__ = [
    "FlowerClient",
    "FLUXRouter",
    "plot_flux_embeddings",
]
