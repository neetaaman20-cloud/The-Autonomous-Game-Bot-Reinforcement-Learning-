import os
import csv
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, "run_" + timestamp + ".csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "score", "mean_score", "epsilon", "reward"])
        print("Logging to " + self.csv_path)
    def log(self, episode, score, mean_score, epsilon, reward):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, score, round(mean_score, 3), round(epsilon, 4), round(reward, 2)])
