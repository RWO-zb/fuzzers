import sys
import os
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../qdfuzz')))

from hybridfuzz.strategy_base import FuzzingStrategy

try:
    from qdfuzz.common import compute_cell
    from qdfuzz.bw_common import get_edges
    QDFUZZ_IMPORTED = True
except ImportError:
    QDFUZZ_IMPORTED = False

class QDFuzzAdapter(FuzzingStrategy):
    name = "qdfuzz"

    def __init__(self):
        self.cells = []
        self.xedges = None
        self.yedges = None
        self.cell_counts = {}

    def initialize(self, config):
        if QDFUZZ_IMPORTED:
            # Descriptors [6,7] usually refer to mean X and mean Y or similar behavior descriptors
            cwd = os.getcwd()
            qdfuzz_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../qdfuzz'))
            try:
                os.chdir(qdfuzz_dir)
                self.xedges, self.yedges = get_edges(config.seed, [6, 7])
            except Exception as e:
                print(f"[QDFuzzAdapter] Error loading edges: {e}")
                self.xedges, self.yedges = None, None
            finally:
                os.chdir(cwd)

    def mutate_or_generate(self, seed):
        if seed is None or seed.get("testcase") is None:
            return np.random.randint(low=1, high=4, size=15)
            
        testcase = seed["testcase"]
        
        mutation = np.random.choice(2, 15, p=[0.9, 0.1])
        if np.sum(mutation) == 0:
            mutation[0] = 1
        mutated = testcase + mutation
        mutated = np.remainder(mutated, 4)
        return np.clip(mutated, 1, 3)

    def update(self, candidate, result, features):
        if not QDFUZZ_IMPORTED or not features.get("behavior_descriptor"):
            return
            
        behavior = np.array(features["behavior_descriptor"])
        if len(behavior) > 7:
            extracted_behavior = np.array([behavior[6], behavior[7]])
            try:
                cell = tuple(compute_cell(extracted_behavior, self.xedges, self.yedges).tolist())
                if cell not in self.cells:
                    self.cells.append(cell)
                self.cell_counts[cell] = self.cell_counts.get(cell, 0) + 1
            except Exception:
                pass

    def compute_feedback(self, candidate, result, features):
        scores = {}
        if not QDFUZZ_IMPORTED or not features.get("behavior_descriptor"):
            return scores
            
        behavior = np.array(features["behavior_descriptor"])
        if len(behavior) > 7:
            extracted_behavior = np.array([behavior[6], behavior[7]])
            try:
                cell = tuple(compute_cell(extracted_behavior, self.xedges, self.yedges).tolist())
                # Novelty based on cell count (fewer counts -> higher novelty)
                count = self.cell_counts.get(cell, 0)
                # Compute exponential decay for novelty score to map into [0, 1]
                scores["diversity_score"] = math.exp(-count)
            except Exception:
                scores["diversity_score"] = 0.0
                
        return scores
