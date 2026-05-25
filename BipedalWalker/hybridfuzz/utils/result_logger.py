import os
import json
import csv
import time
import numpy as np


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    return value

class ResultLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "crashes"), exist_ok=True)
        
        self.iteration_log_path = os.path.join(output_dir, "iteration_log.csv")
        self.summary_path = os.path.join(output_dir, "summary.json")
        self.strategy_stats_path = os.path.join(output_dir, "strategy_stats.csv")
        self.seed_pool_stats_path = os.path.join(output_dir, "seed_pool_stats.csv")
        self.fault_lineage_path = os.path.join(output_dir, "fault_lineage.jsonl")
        
        # Initialize CSV files with headers
        with open(self.iteration_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "selected_strategy", "seed_id", "candidate_id",
                "candidate",
                "parent_seed_id", "root_seed_id", "selected_seed_source_strategy",
                "selected_seed_root_strategy", "mutation_depth", "is_cross_strategy_reuse",
                "is_fault", "did_physical_crash", "is_reward_fault", "is_unique_crash", "crash_signature",
                "episode_reward",
                "novelty_score", "diversity_score", "uncertainty_score",
                "g_model_score", "scheduler_reward", "execution_cost", "seed_pool_size", "survival_steps",
                "qd_behavior", "qd_cell", "qd_new_cell", "qd_cell_count",
                "reward_drop_score", "raw_reward_drop", "raw_seq_coverage", "seq_tapnet_prediction",
                "g_model_novelty", "g_model_abstract_id", "selected_strategy_scores"
            ])
            
        with open(self.strategy_stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["strategy", "selected_count", "total_reward", "average_reward"])
            
        with open(self.seed_pool_stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "pool_size", "total_faults"])
            
        self.start_time = time.time()
        self.total_crashes = 0
        self.physical_crashes = 0
        self.reward_faults = 0
        self.unique_crashes = 0
        self.time_to_first_crash = None
        self.cross_strategy_reuse_count = 0
        self.faults_from_cross_strategy_reuse = 0
        self.unique_faults_from_cross_strategy_reuse = 0
        self.per_strategy_seed_contribution = {}
        self.per_strategy_fault_contribution = {}

    def log_iteration(self, iteration, strategy_name, seed_id, candidate_id, candidate, features, scheduler_reward, pool_size, episode_reward):
        is_cross = bool(features.get("is_cross_strategy_reuse", False))
        if is_cross:
            self.cross_strategy_reuse_count += 1

        source = features.get("source_strategy_for_added_seed", strategy_name)
        self.per_strategy_seed_contribution[source] = self.per_strategy_seed_contribution.get(source, 0) + 1

        if features["is_crash"]:
            self.total_crashes += 1
            self.per_strategy_fault_contribution[strategy_name] = self.per_strategy_fault_contribution.get(strategy_name, 0) + 1
            if is_cross:
                self.faults_from_cross_strategy_reuse += 1
            if self.time_to_first_crash is None:
                self.time_to_first_crash = time.time() - self.start_time
            if features["is_unique_crash"]:
                self.unique_crashes += 1
                if is_cross:
                    self.unique_faults_from_cross_strategy_reuse += 1
        if features.get("did_physical_crash", False):
            self.physical_crashes += 1
        if features.get("is_reward_fault", False):
            self.reward_faults += 1

        with open(self.iteration_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, strategy_name, seed_id, candidate_id,
                json.dumps(json_safe(candidate)),
                features.get("parent_seed_id"),
                features.get("root_seed_id"),
                features.get("selected_seed_source_strategy"),
                features.get("selected_seed_root_strategy"),
                features.get("mutation_depth"),
                is_cross,
                features["is_crash"], features.get("did_physical_crash", False), features.get("is_reward_fault", False),
                features["is_unique_crash"], features["crash_signature"],
                episode_reward,
                features["novelty_score"], features["diversity_score"], features["uncertainty_score"],
                features["g_model_score"], scheduler_reward, features["execution_cost"], pool_size, features.get("survival_steps", ""),
                json.dumps(json_safe(features.get("qd_behavior", []))),
                json.dumps(json_safe(features.get("qd_cell", None))),
                features.get("qd_new_cell", ""),
                features.get("qd_cell_count", ""),
                features.get("reward_drop_score", ""),
                features.get("raw_reward_drop", ""),
                features.get("raw_seq_coverage", ""),
                features.get("seq_tapnet_prediction", ""),
                features.get("g_model_novelty", ""),
                features.get("g_model_abstract_id", ""),
                json.dumps(json_safe(features.get("selected_strategy_scores", {})))
            ])

        if features["is_crash"]:
            fault_payload = {
                "iteration": iteration,
                "candidate": json_safe(candidate),
                "fault_type": "physical_crash" if features.get("did_physical_crash", False) else "reward_fault",
                "selected_strategy": strategy_name,
                "parent_seed_id": features.get("parent_seed_id"),
                "parent_source_strategy": features.get("selected_seed_source_strategy"),
                "root_seed_id": features.get("root_seed_id"),
                "root_source_strategy": features.get("selected_seed_root_strategy"),
                "mutation_depth": features.get("mutation_depth"),
                "is_cross_strategy_reuse": is_cross,
                "episode_reward": episode_reward,
                "survival_steps": features.get("survival_steps"),
                "qd_cell": json_safe(features.get("qd_cell")),
                "crash_signature": features.get("crash_signature"),
            }
            with open(self.fault_lineage_path, 'a') as f:
                f.write(json.dumps(fault_payload) + "\n")
            
        with open(self.seed_pool_stats_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, pool_size, self.total_crashes])

    def save_config(self, config):
        with open(os.path.join(self.output_dir, "config.json"), 'w') as f:
            json.dump(vars(config), f, indent=4)

    def log_summary(self, scheduler, final_pool_size, adapter_status=None, seed_pool_stats=None):
        total_execution_time = time.time() - self.start_time
        
        per_strategy_selected_count = {s: scheduler.counts[s] for s in scheduler.strategies}
        per_strategy_average_reward = {
            s: scheduler.rewards[s] / max(1, scheduler.counts[s]) 
            for s in scheduler.strategies
        }

        summary = {
            "total_faults": self.total_crashes,
            "physical_crashes": self.physical_crashes,
            "reward_faults": self.reward_faults,
            "unique_crashes": self.unique_crashes,
            "time_to_first_crash": self.time_to_first_crash,
            "final_seed_pool_size": final_pool_size,
            "total_execution_time": total_execution_time,
            "per_strategy_selected_count": per_strategy_selected_count,
            "per_strategy_average_reward": per_strategy_average_reward,
            "adapter_status": adapter_status or {},
            "seed_pool_stats": seed_pool_stats or {},
            "cross_strategy_reuse_count": self.cross_strategy_reuse_count,
            "cross_strategy_reuse_rate": self.cross_strategy_reuse_count / max(1, sum(per_strategy_selected_count.values())),
            "faults_from_cross_strategy_reuse": self.faults_from_cross_strategy_reuse,
            "unique_faults_from_cross_strategy_reuse": self.unique_faults_from_cross_strategy_reuse,
            "per_strategy_seed_contribution": self.per_strategy_seed_contribution,
            "per_strategy_fault_contribution": self.per_strategy_fault_contribution,
        }
        
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        with open(self.strategy_stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["strategy", "selected_count", "total_reward", "average_reward"])
            for s in scheduler.strategies:
                writer.writerow([s, scheduler.counts[s], scheduler.rewards[s], per_strategy_average_reward[s]])
