# logger.py
# no matplotlib. just logs to csv + keeps running averages in memory.

import csv
import os


class LiveLogger:
    def __init__(self, csv_path="metrics.csv"):
        self.csv_path = csv_path
        self.header_written = os.path.exists(csv_path)

    def log(self, step, train_reward, eval_reward, loss_actor, loss_critic):
        row = {
            "step": int(step),
            "train_reward": float(train_reward),
            "eval_reward": float(eval_reward),
            "loss_actor": float(loss_actor),
            "loss_critic": float(loss_critic),
        }

        write_header = not self.header_written
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                w.writeheader()
                self.header_written = True
            w.writerow(row)
