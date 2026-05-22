import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../g-model')))

from hybridfuzz.strategy_base import FuzzingStrategy

try:
    from g_model.diffusion import Diffusion
    from g_model.interfaces import Memory, Density, Grid
    GMODEL_IMPORTED = True
except ImportError:
    try:
        from diffusion import Diffusion
        from interfaces import Memory, Density, Grid
        GMODEL_IMPORTED = True
    except ImportError:
        GMODEL_IMPORTED = False

class GModelAdapter(FuzzingStrategy):
    name = "g-model"

    def __init__(self):
        self.diffusion = None
        self.memory = None
        self.cases_buffer = []

    def initialize(self, config):
        if GMODEL_IMPORTED:
            # BipedalWalker testcase dimension is 15
            self.diffusion = Diffusion(batch_size=1, epoch=10, data_size=15, training_step_per_spoch=10, num_diffusion_step=25)
            self.diffusion.setup()
            self.memory = Memory(size=100)

    def mutate_or_generate(self, seed):
        if self.diffusion and np.random.rand() < 0.5:
            # 50% chance to use diffusion model to generate a brand new candidate
            try:
                candidate = self.diffusion.generate()
                # Discretize it to valid values [1, 2, 3] like other fuzzers
                candidate = np.round(candidate)
                candidate = np.clip(candidate, 1, 3)
                return candidate.astype(int)
            except Exception:
                pass
                
        # Fallback to standard mutation
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
        if not GMODEL_IMPORTED:
            return
            
        # Add to buffer, retrain diffusion periodically
        self.cases_buffer.append(candidate)
        if len(self.cases_buffer) >= 50:
            try:
                # Train the generative model using collected cases
                self.diffusion.train(np.array(self.cases_buffer), None, 'generative')
            except Exception:
                pass
            self.cases_buffer = []

    def compute_feedback(self, candidate, result, features):
        scores = {}
        # G-model focuses on generative diversity. 
        # Here we assign a generic score or rely on the shared pool to combine.
        if GMODEL_IMPORTED and self.memory:
            # A mock g_model_score indicating density/novelty combined
            scores["g_model_score"] = 0.5 # A placeholder; in a full implementation, density/sensitivity is queried.
        return scores
