import time
import numpy as np
from .crash_utils import generate_crash_signature

class FeatureExtractor:
    def __init__(self):
        self.seen_crashes = set()

    def extract_features(self, env, obs_seq, total_reward, is_crash, execution_cost, adapter_scores=None, execution_result=None):
        """
        Extract features from the episode execution.
        obs_seq: sequence of observations
        total_reward: accumulated reward
        is_crash: whether the episode resulted in a crash
        execution_cost: time or steps taken
        adapter_scores: dict containing specific scores (novelty, diversity, etc.) computed by adapters or models
        """
        if adapter_scores is None:
            adapter_scores = {}

        did_physical_crash = bool(is_crash)
        is_reward_fault = False
        survival_steps = len(obs_seq)
        behavior_features = []
        qd_behavior = []
        if execution_result is not None:
            did_physical_crash = bool(execution_result.get("did_physical_crash", False))
            is_reward_fault = bool(execution_result.get("is_reward_fault", False))
            is_crash = bool(execution_result.get("is_fault", is_crash))
            survival_steps = int(execution_result.get("survival_steps", survival_steps))
            behavior_features = execution_result.get("behavior_features", [])
            qd_behavior = execution_result.get("qd_behavior", [])

        obs_seq_arr = np.array(obs_seq)
        if len(obs_seq_arr) > 0:
            behavior_descriptor = behavior_features if behavior_features else np.mean(obs_seq_arr, axis=0).tolist()
            # Trajectory signature could be downsampled sequence
            indices = np.linspace(0, len(obs_seq_arr)-1, min(10, len(obs_seq_arr)), dtype=int)
            trajectory_signature = obs_seq_arr[indices].flatten().tolist()
            
            crash_signature = generate_crash_signature(obs_seq) if is_crash else None
        else:
            behavior_descriptor = []
            trajectory_signature = []
            crash_signature = None

        is_unique_crash = False
        if is_crash and crash_signature is not None:
            if crash_signature not in self.seen_crashes:
                is_unique_crash = True
                self.seen_crashes.add(crash_signature)

        return {
            "is_crash": is_crash,
            "is_fault": is_crash,
            "did_physical_crash": did_physical_crash,
            "is_reward_fault": is_reward_fault,
            "is_unique_crash": is_unique_crash,
            "crash_signature": crash_signature,
            "trajectory_signature": trajectory_signature,
            "behavior_descriptor": behavior_descriptor,
            "behavior_features": behavior_features,
            "qd_behavior": qd_behavior,
            "survival_steps": survival_steps,
            "novelty_score": adapter_scores.get("novelty_score", 0.0),
            "diversity_score": adapter_scores.get("diversity_score", 0.0),
            "uncertainty_score": adapter_scores.get("uncertainty_score", 0.0),
            "g_model_score": adapter_scores.get("g_model_score", 0.0),
            "execution_cost": execution_cost,
        }
