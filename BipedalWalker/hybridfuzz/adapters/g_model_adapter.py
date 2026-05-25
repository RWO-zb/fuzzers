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
        self.imported = GMODEL_IMPORTED
        self.diffusion_train_count = 0
        self.generate_success_count = 0
        self.generate_exception_count = 0
        self.method = "generative+novelty"
        self.train_step = 50
        self.novelty_grid = None
        self.novelty_dict = {}
        self.metric_buffer = []
        self.last_abstract_id = None
        self.last_novelty = None

    def initialize(self, config):
        self.method = getattr(config, "g_model_method", "generative+novelty")
        self.train_step = getattr(config, "g_model_train_step", 50)
        if GMODEL_IMPORTED:
            # BipedalWalker testcase dimension is 15
            self.diffusion = Diffusion(batch_size=1, epoch=10, data_size=15, training_step_per_spoch=10, num_diffusion_step=25)
            self.diffusion.setup()
            self.memory = Memory(size=100)
            min_obs = np.array([-5 for _ in range(24)])
            max_obs = np.array([5 for _ in range(24)])
            self.novelty_grid = Grid(min_obs, max_obs, getattr(config, "g_model_grid", 5))

    def _compute_novelty(self, result):
        if self.novelty_grid is None:
            return 0.0, None, 0
        final_state = result.get("final_state", [])
        if len(final_state) == 0:
            return 0.0, None, 0
        final_state = np.asarray(final_state, dtype=float).reshape(1, -1)
        abstract_id = self.novelty_grid.state_abstract(final_state)[0]
        next_count = self.novelty_dict.get(abstract_id, 0) + 1
        novelty = float(np.exp(-(next_count - 1)))
        return novelty, abstract_id, next_count

    def mutate_or_generate(self, seed):
        if self.diffusion and np.random.rand() < 0.5:
            # 50% chance to use diffusion model to generate a brand new candidate
            try:
                candidate = self.diffusion.generate()
                # Discretize it to valid values [1, 2, 3] like other fuzzers
                candidate = np.round(candidate)
                candidate = np.clip(candidate, 1, 3)
                self.generate_success_count += 1
                return candidate.astype(int)
            except Exception:
                self.generate_exception_count += 1
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
            
        candidate_arr = np.asarray(candidate, dtype=int)
        self.cases_buffer.append(candidate_arr)
        novelty = float(features.get("g_model_novelty", features.get("g_model_score", 0.0)))
        abstract_id = features.get("g_model_abstract_id")
        if abstract_id is not None:
            self.novelty_dict[abstract_id] = self.novelty_dict.get(abstract_id, 0) + 1
        self.metric_buffer.append([novelty])

        if self.memory is not None:
            reward = float(result.get("reward", 0.0))
            self.memory.append(candidate_arr, 0.0, 0.0, reward, novelty)

        if len(self.cases_buffer) >= self.train_step:
            try:
                self.diffusion.train(np.array(self.cases_buffer), np.array(self.metric_buffer), self.method)
                self.diffusion_train_count += 1
            except Exception:
                pass
            self.cases_buffer = []
            self.metric_buffer = []

    def compute_feedback(self, candidate, result, features):
        scores = {}
        if GMODEL_IMPORTED:
            novelty, abstract_id, next_count = self._compute_novelty(result)
            self.last_novelty = novelty
            self.last_abstract_id = abstract_id
            scores["g_model_score"] = novelty
            scores["g_model_novelty"] = novelty
            scores["g_model_abstract_id"] = abstract_id
            scores["g_model_abstract_count"] = next_count
        return scores

    def get_status(self):
        return {
            "imported": self.imported,
            "initialized": self.diffusion is not None,
            "memory_size": self.memory.get_index() if self.memory is not None else 0,
            "cases_buffer_size": len(self.cases_buffer),
            "diffusion_train_count": self.diffusion_train_count,
            "generate_success_count": self.generate_success_count,
            "generate_exception_count": self.generate_exception_count,
            "method": self.method,
            "train_step": self.train_step,
            "novelty_cells": len(self.novelty_dict),
            "last_abstract_id": self.last_abstract_id,
            "last_novelty": self.last_novelty,
        }
