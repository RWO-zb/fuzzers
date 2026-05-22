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

    def initialize(self, config):
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
        pass # RND is updated in compute_feedback

    def compute_feedback(self, candidate, result, features):
        scores = {}
        if self.fuzzer and hasattr(self.fuzzer, 'train_rnd'):
            # Assuming result["obs_seq"] contains the trajectory
            obs_seq = result.get("obs_seq", [])
            if len(obs_seq) > 0:
                try:
                    intrinsic_reward = self.fuzzer.train_rnd(obs_seq)
                    # Normalize intrinsic reward as uncertainty score
                    scores["uncertainty_score"] = float(intrinsic_reward)
                except Exception:
                    scores["uncertainty_score"] = 0.0
        return scores
