import os
import json
import csv
import time

class ResultLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "crashes"), exist_ok=True)
        
        self.iteration_log_path = os.path.join(output_dir, "iteration_log.csv")
        self.summary_path = os.path.join(output_dir, "summary.json")
        self.strategy_stats_path = os.path.join(output_dir, "strategy_stats.csv")
        self.seed_pool_stats_path = os.path.join(output_dir, "seed_pool_stats.csv")
        
        # Initialize CSV files with headers
        with open(self.iteration_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "selected_strategy", "seed_id", "candidate_id",
                "is_crash", "is_unique_crash", "crash_signature",
                "novelty_score", "diversity_score", "uncertainty_score",
                "g_model_score", "execution_cost", "reward", "seed_pool_size"
            ])
            
        with open(self.strategy_stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["strategy", "selected_count", "total_reward", "average_reward"])
            
        with open(self.seed_pool_stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "pool_size", "total_crashes"])
            
        self.start_time = time.time()
        self.total_crashes = 0
        self.unique_crashes = 0
        self.time_to_first_crash = None

    def log_iteration(self, iteration, strategy_name, seed_id, candidate_id, features, reward, pool_size):
        if features["is_crash"]:
            self.total_crashes += 1
            if self.time_to_first_crash is None:
                self.time_to_first_crash = time.time() - self.start_time
            if features["is_unique_crash"]:
                self.unique_crashes += 1

        with open(self.iteration_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, strategy_name, seed_id, candidate_id,
                features["is_crash"], features["is_unique_crash"], features["crash_signature"],
                features["novelty_score"], features["diversity_score"], features["uncertainty_score"],
                features["g_model_score"], features["execution_cost"], reward, pool_size
            ])
            
        with open(self.seed_pool_stats_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, pool_size, self.total_crashes])

    def save_config(self, config):
        with open(os.path.join(self.output_dir, "config.json"), 'w') as f:
            json.dump(vars(config), f, indent=4)

    def log_summary(self, scheduler, final_pool_size):
        total_execution_time = time.time() - self.start_time
        
        per_strategy_selected_count = {s: scheduler.counts[s] for s in scheduler.strategies}
        per_strategy_average_reward = {
            s: scheduler.rewards[s] / max(1, scheduler.counts[s]) 
            for s in scheduler.strategies
        }

        summary = {
            "total_crashes": self.total_crashes,
            "unique_crashes": self.unique_crashes,
            "time_to_first_crash": self.time_to_first_crash,
            "final_seed_pool_size": final_pool_size,
            "total_execution_time": total_execution_time,
            "per_strategy_selected_count": per_strategy_selected_count,
            "per_strategy_average_reward": per_strategy_average_reward
        }
        
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        with open(self.strategy_stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["strategy", "selected_count", "total_reward", "average_reward"])
            for s in scheduler.strategies:
                writer.writerow([s, scheduler.counts[s], scheduler.rewards[s], per_strategy_average_reward[s]])
