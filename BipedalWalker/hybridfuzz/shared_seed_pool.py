import json
import uuid
import time
import numpy as np

class SharedSeedPool:
    def __init__(self, config):
        self.config = config
        self.pool = {}
    
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
        seed_id = str(uuid.uuid4())
        seed_data = {
            "seed_id": seed_id,
            "testcase": testcase,
            "crash_score": kwargs.get("crash_score", 0.0),
            "novelty_score": kwargs.get("novelty_score", 0.0),
            "diversity_score": kwargs.get("diversity_score", 0.0),
            "uncertainty_score": kwargs.get("uncertainty_score", 0.0),
            "g_model_score": kwargs.get("g_model_score", 0.0),
            "execution_cost": kwargs.get("execution_cost", 0.0),
            "trajectory_signature": kwargs.get("trajectory_signature", None),
            "behavior_descriptor": kwargs.get("behavior_descriptor", None),
            "crash_signature": kwargs.get("crash_signature", None),
            "source_strategy": source_strategy,
            "created_at": int(time.time()),
            "num_mutations": kwargs.get("num_mutations", 0),
        }
        self.pool[seed_id] = seed_data
        return seed_id

    def select_seed(self):
        if not self.pool:
            return None
            
        # Roulette wheel selection based on unified score
        scores = []
        seed_ids = []
        for sid, data in self.pool.items():
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
            "pool_size": len(self.pool)
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
