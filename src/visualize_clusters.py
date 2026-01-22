import numpy as np
import matplotlib.pyplot as plt

def plot_flux_embeddings(descriptor_store, cluster_assignments, out_path="flux_clusters.png"):
    """
    Simple 2D projection using PCA (no extra deps), so it works in Colab reliably.
    descriptor_store: {client_id: 1D vector}
    cluster_assignments: {client_id: cluster_id}
    """
    if len(descriptor_store) < 2:
        print("[plot_flux_embeddings] Not enough descriptors to plot.")
        return

    cids = sorted(descriptor_store.keys())
    X = np.stack([descriptor_store[c] for c in cids])  # [N, D]
    labels = np.array([cluster_assignments.get(c, -1) for c in cids])

    # PCA to 2D
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    X2 = Xc @ Vt[:2].T

    plt.figure(figsize=(7, 6))
    for cl in np.unique(labels):
        pts = X2[labels == cl]
        plt.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {cl}", s=60)

    for i, cid in enumerate(cids):
        plt.text(X2[i, 0], X2[i, 1], str(cid), fontsize=9)

    plt.title("AdaFLUX Client Clusters in Descriptor Space (PCA)")
    plt.legend()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot_flux_embeddings] Saved: {out_path}")
