import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

def plot_flux_embeddings(descriptor_store, cluster_assignments, out_path="flux_clusters.png"):
    X = np.stack(list(descriptor_store.values()))
    labels = np.array([cluster_assignments[cid] for cid in sorted(descriptor_store.keys())])

    reducer = UMAP(n_neighbors=5, min_dist=0.1, metric="euclidean")
    X2d = reducer.fit_transform(X)

    plt.figure(figsize=(7,6))
    for cl in np.unique(labels):
        pts = X2d[labels == cl]
        plt.scatter(pts[:,0], pts[:,1], label=f"Cluster {cl}", s=60)

    for i, cid in enumerate(sorted(descriptor_store.keys())):
        plt.text(X2d[i,0], X2d[i,1], str(cid), fontsize=9)

    plt.title("AdaFLUX Client Clusters in Descriptor Space")
    plt.legend()
    plt.savefig(out_path, dpi=300)
    plt.close()
