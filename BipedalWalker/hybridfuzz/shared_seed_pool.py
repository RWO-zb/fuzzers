import json
import uuid
import time
import numpy as np

class SharedSeedPool:
    def __init__(self, config):
        self.config = config
        self.pool = {}
        self.testcase_index = {}
        self.duplicate_discards = 0
        self.evictions = 0

    def _testcase_key(self, testcase):
        arr = np.asarray(testcase, dtype=int).reshape(-1)
        return ",".join(map(str, arr.tolist()))
    
    def _compute_score(self, seed_data):
        return (
            self.config.alpha * seed_data.get("crash_score", 0.0) +
            self.config.beta * seed_data.get("novelty_score", 0.0) +
            self.config.gamma * seed_data.get("diversity_score", 0.0) +
            self.config.delta * seed_data.get("uncertainty_score", 0.0) +
            self.config.eta * seed_data.get("g_model_score", 0.0) -
            self.config.lambda_cost * seed_data.get("execution_cost", 0.0)
        )

    def add_seed(self, testcase, source_strategy, **kwargs):
        testcase_key = self._testcase_key(testcase)
        if testcase_key in self.testcase_index:
            self.duplicate_discards += 1
            return self.testcase_index[testcase_key]

        seed_id = str(uuid.uuid4())
        seed_data = {
            "seed_id": seed_id,
            "testcase": testcase,
            "testcase_key": testcase_key,
            "crash_score": kwargs.get("crash_score", 0.0),
            "did_physical_crash": kwargs.get("did_physical_crash", False),
            "is_reward_fault": kwargs.get("is_reward_fault", False),
            "novelty_score": kwargs.get("novelty_score", 0.0),
            "diversity_score": kwargs.get("diversity_score", 0.0),
            "uncertainty_score": kwargs.get("uncertainty_score", 0.0),
            "g_model_score": kwargs.get("g_model_score", 0.0),
            "execution_cost": kwargs.get("execution_cost", 0.0),
            "trajectory_signature": kwargs.get("trajectory_signature", None),
            "behavior_descriptor": kwargs.get("behavior_descriptor", None),
            "behavior_features": kwargs.get("behavior_features", None),
            "qd_behavior": kwargs.get("qd_behavior", None),
            "qd_cell": kwargs.get("qd_cell", None),
            "crash_signature": kwargs.get("crash_signature", None),
            "reward": kwargs.get("reward", None),
            "final_state": kwargs.get("final_state", None),
            "survival_steps": kwargs.get("survival_steps", None),
            "source_strategy": source_strategy,
            "parent_seed_id": kwargs.get("parent_seed_id", None),
            "parent_source_strategy": kwargs.get("parent_source_strategy", None),
            "root_seed_id": kwargs.get("root_seed_id", None),
            "root_source_strategy": kwargs.get("root_source_strategy", source_strategy),
            "created_at": int(time.time()),
            "num_mutations": kwargs.get("num_mutations", 0),
        }
        if seed_data["root_seed_id"] is None:
            seed_data["root_seed_id"] = seed_id
        self.pool[seed_id] = seed_data
        self.testcase_index[testcase_key] = seed_id
        self._evict_if_needed()
        return seed_id

    def _evict_if_needed(self):
        max_pool_size = getattr(self.config, "max_pool_size", 0)
        if max_pool_size is None or max_pool_size <= 0:
            return

        while len(self.pool) > max_pool_size:
            victim_id = min(self.pool, key=lambda sid: self._compute_score(self.pool[sid]))
            victim = self.pool.pop(victim_id)
            self.testcase_index.pop(victim.get("testcase_key"), None)
            self.evictions += 1

    def select_seed(self, source_strategy=None):
        candidates = self.pool
        if source_strategy is not None:
            candidates = {sid: data for sid, data in self.pool.items() if data.get("source_strategy") == source_strategy}
            if not candidates:
                return None

        if not candidates:
            return None
            
        # Roulette wheel selection based on unified score
        scores = []
        seed_ids = []
        for sid, data in candidates.items():
            s = self._compute_score(data)
            scores.append(s)
            seed_ids.append(sid)
            
        # Shift scores to be positive
        scores = np.array(scores)
        min_score = np.min(scores)
        if min_score < 0:
            scores = scores - min_score
        
        # Add a small epsilon to avoid zero probabilities
        scores = scores + 1e-5
        
        probs = scores / np.sum(scores)
        selected_id = np.random.choice(seed_ids, p=probs)
        return self.pool[selected_id]

    def update_seed(self, seed_id, **kwargs):
        if seed_id in self.pool:
            for k, v in kwargs.items():
                if k in self.pool[seed_id]:
                    self.pool[seed_id][k] = v

    def get_statistics(self):
        return {
            "pool_size": len(self.pool),
            "duplicate_discards": self.duplicate_discards,
            "evictions": self.evictions,
        }

    def save(self, filepath):
        # Convert numpy arrays to lists for JSON serialization
        save_data = {}
        for k, v in self.pool.items():
            save_data[k] = v.copy()
            if isinstance(v["testcase"], np.ndarray):
                save_data[k]["testcase"] = v["testcase"].tolist()
                
        with open(filepath, 'w') as f:
            json.dump(save_data, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            load_data = json.load(f)
            
        for k, v in load_data.items():
            if isinstance(v["testcase"], list):
                v["testcase"] = np.array(v["testcase"])
            self.pool[k] = v
            self.testcase_index[v.get("testcase_key", self._testcase_key(v["testcase"]))] = k
