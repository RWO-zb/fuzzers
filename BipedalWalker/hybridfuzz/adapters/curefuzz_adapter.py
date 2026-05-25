import sys
import os
import numpy as np

# Append paths to allow importing from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../curefuzz')))

from hybridfuzz.strategy_base import FuzzingStrategy

try:
    from curefuzz.fuzz.cure_fuzz import CureFuzz
    CURE_IMPORTED = True
except ImportError:
    CURE_IMPORTED = False

class CureFuzzAdapter(FuzzingStrategy):
    name = "curefuzz"

    def __init__(self):
        self.fuzzer = None
        self.raw_uncertainty_min = None
        self.raw_uncertainty_max = None
        self.rnd_success_count = 0
        self.rnd_exception_count = 0
        self.imported = CURE_IMPORTED
        self.norm_mode = "rolling"

    def _normalize_uncertainty(self, value):
        value = float(np.asarray(value).reshape(-1)[0])
        if self.raw_uncertainty_min is None:
            self.raw_uncertainty_min = value
            self.raw_uncertainty_max = value
            return 1.0

        self.raw_uncertainty_min = min(self.raw_uncertainty_min, value)
        self.raw_uncertainty_max = max(self.raw_uncertainty_max, value)

        if self.norm_mode == "log":
            return float(np.clip(np.log1p(max(0.0, value)) / 10.0, 0.0, 1.0))

        denom = self.raw_uncertainty_max - self.raw_uncertainty_min
        if denom <= 1e-9:
            return 1.0
        return float(np.clip((value - self.raw_uncertainty_min) / denom, 0.0, 1.0))

    def initialize(self, config):
        self.norm_mode = getattr(config, "uncertainty_norm", "rolling")
        if CURE_IMPORTED:
            self.fuzzer = CureFuzz()
        else:
            self.fuzzer = None
            print("[Warning] CureFuzz could not be fully imported. Using fallback.")

    def mutate_or_generate(self, seed):
        if seed is None or seed.get("testcase") is None:
            return np.random.randint(low=1, high=4, size=15)
            
        testcase = seed["testcase"]
        if self.fuzzer and hasattr(self.fuzzer, 'mutation'):
            try:
                return self.fuzzer.mutation(testcase)
            except Exception:
                pass
        
        # Fallback mutation
        mutation = np.random.choice(2, 15, p=[0.9, 0.1])
        if np.sum(mutation) == 0:
            mutation[0] = 1
        mutated = testcase + mutation
        mutated = np.remainder(mutated, 4)
        return np.clip(mutated, 1, 3)

    def update(self, candidate, result, features):
        if self.fuzzer is None:
            return
        if result.get("is_fault", False) and hasattr(self.fuzzer, "add_crash"):
            try:
                self.fuzzer.add_crash(candidate)
            except Exception:
                pass
        elif hasattr(self.fuzzer, "further_mutation"):
            try:
                entropy = 0.0
                final_state = result.get("final_state", [])
                parent_seed = features.get("parent_seed")
                if parent_seed and parent_seed.get("final_state") is not None:
                    entropy = float(np.linalg.norm(np.asarray(final_state) - np.asarray(parent_seed["final_state"])))
                self.fuzzer.further_mutation(candidate, result.get("reward", 0.0), entropy, features.get("uncertainty_score", 0.0), final_state, candidate)
            except Exception:
                pass

    def compute_feedback(self, candidate, result, features):
        scores = {}
        if self.fuzzer and hasattr(self.fuzzer, 'train_rnd'):
            # Assuming result["obs_seq"] contains the trajectory
            obs_seq = result.get("obs_seq", [])
            if len(obs_seq) > 0:
                try:
                    intrinsic_reward = self.fuzzer.train_rnd(obs_seq)
                    raw_value = float(np.asarray(intrinsic_reward).reshape(-1)[0])
                    scores["uncertainty_score"] = self._normalize_uncertainty(raw_value)
                    scores["raw_uncertainty_score"] = raw_value
                    self.rnd_success_count += 1
                except Exception:
                    scores["uncertainty_score"] = 0.0
                    self.rnd_exception_count += 1
        return scores

    def get_status(self):
        return {
            "imported": self.imported,
            "initialized": self.fuzzer is not None,
            "rnd_success_count": self.rnd_success_count,
            "rnd_exception_count": self.rnd_exception_count,
            "raw_uncertainty_min": self.raw_uncertainty_min,
            "raw_uncertainty_max": self.raw_uncertainty_max,
            "norm_mode": self.norm_mode,
        }
