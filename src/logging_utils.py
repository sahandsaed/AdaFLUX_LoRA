import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:
    def __init__(self, run_name="adaflux_run", base_dir="runs"):
        os.makedirs(base_dir, exist_ok=True)
        self.path = os.path.join(base_dir, run_name)
        self.writer = SummaryWriter(self.path)

    def log_round_metrics(self, round_idx, loss, acc, cluster_summary=None):
        self.writer.add_scalar("Loss/Validation", loss, round_idx)
        self.writer.add_scalar("Accuracy/Validation", acc, round_idx)

        if cluster_summary:
            for cid, size in cluster_summary.items():
                self.writer.add_scalar(f"Cluster/{cid}_size", size, round_idx)

    def close(self):
        self.writer.close()
