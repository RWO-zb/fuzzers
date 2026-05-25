import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mdpfuzz')))

from hybridfuzz.strategy_base import FuzzingStrategy

class MDPFuzzAdapter(FuzzingStrategy):
    name = "mdpfuzz"

    def __init__(self):
        self.previous_rewards = {}
        self.feedback_count = 0
        self.reward_drop_count = 0
        self.reward_drop_min = None
        self.reward_drop_max = None
        self.reward_drop_scale = 20.0

    def _normalize_reward_drop(self, value):
        value = max(0.0, float(value))
        if self.reward_drop_min is None:
            self.reward_drop_min = value
            self.reward_drop_max = value
        else:
            self.reward_drop_min = min(self.reward_drop_min, value)
            self.reward_drop_max = max(self.reward_drop_max, value)
        return float(1.0 - np.exp(-value / max(self.reward_drop_scale, 1e-12)))

    def initialize(self, config):
        self.reward_drop_scale = getattr(config, "reward_drop_scale", 20.0)

    def _old_normalize_reward_drop(self, value):
        value = max(0.0, float(value))
        if self.reward_drop_min is None:
            self.reward_drop_min = value
            self.reward_drop_max = value
            return 1.0 if value > 0 else 0.0
        self.reward_drop_min = min(self.reward_drop_min, value)
        self.reward_drop_max = max(self.reward_drop_max, value)
        denom = self.reward_drop_max - self.reward_drop_min
        if denom <= 1e-12:
            return 1.0 if value > 0 else 0.0
        return float(np.clip((value - self.reward_drop_min) / denom, 0.0, 1.0))

    def mutate_or_generate(self, seed):
        if seed is None or seed.get("testcase") is None:
            return np.random.randint(low=1, high=4, size=15)
            
        testcase = seed["testcase"]
        
        # Standard MDPFuzz mutation logic
        mutation = np.random.choice(2, 15, p=[0.9, 0.1])
        if np.sum(mutation) == 0:
            mutation[0] = 1
        mutated = testcase + mutation
        mutated = np.remainder(mutated, 4)
        return np.clip(mutated, 1, 3)

    def update(self, candidate, result, features):
        key = tuple(np.asarray(candidate, dtype=int).tolist())
        self.previous_rewards[key] = float(result.get("reward", 0.0))

    def compute_feedback(self, candidate, result, features):
        scores = {}
        is_fault = bool(features.get("is_fault", features.get("is_crash", result.get("is_fault", False))))
        scores["crash_score"] = 1.0 if is_fault else 0.0
        self.feedback_count += 1

        parent_seed = features.get("parent_seed")
        if parent_seed is not None and parent_seed.get("reward") is not None:
            reward_drop = float(parent_seed["reward"]) - float(result.get("reward", 0.0))
            scores["raw_reward_drop"] = max(0.0, reward_drop)
            scores["reward_drop_score"] = self._normalize_reward_drop(reward_drop)
            if reward_drop > 0:
                self.reward_drop_count += 1
        return scores

    def get_status(self):
        return {
            "feedback_count": self.feedback_count,
            "reward_drop_count": self.reward_drop_count,
            "tracked_rewards": len(self.previous_rewards),
            "reward_drop_min": self.reward_drop_min,
            "reward_drop_max": self.reward_drop_max,
            "reward_drop_scale": self.reward_drop_scale,
        }
