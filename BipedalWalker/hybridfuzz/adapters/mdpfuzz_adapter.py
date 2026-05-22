import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mdpfuzz')))

from hybridfuzz.strategy_base import FuzzingStrategy

class MDPFuzzAdapter(FuzzingStrategy):
    name = "mdpfuzz"

    def initialize(self, config):
        pass

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
        pass

    def compute_feedback(self, candidate, result, features):
        # MDPFuzz primarily relies on physical failures and crashes.
        # Returning a basic score based on whether it crashed.
        scores = {}
        if features.get("is_crash", False):
            scores["crash_score"] = 1.0
        else:
            scores["crash_score"] = 0.0
        return scores
