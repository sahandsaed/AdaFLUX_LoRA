import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class FLUXRouter:
    def __init__(self, descriptor_store, cluster_assignments, cluster_models):
        self.descriptor_store = descriptor_store
        self.cluster_assignments = cluster_assignments
        self.cluster_models = cluster_models

    def route(self, descriptor):
        X = np.stack(list(self.descriptor_store.values()))
        d = euclidean_distances([descriptor], X)[0]
        idx = np.argmin(d)
        closest_client = sorted(self.descriptor_store.keys())[idx]
        assigned_cluster = self.cluster_assignments[closest_client]
        return self.cluster_models[assigned_cluster], assigned_cluster
