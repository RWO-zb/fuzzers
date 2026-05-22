import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../seqfuzz')))

from hybridfuzz.strategy_base import FuzzingStrategy

try:
    import torch
    from seqfuzz.tapnet import predict_siamese, Hyperparameter
    SEQFUZZ_IMPORTED = True
except ImportError:
    try:
        import torch
        from tapnet import predict_siamese, Hyperparameter
        SEQFUZZ_IMPORTED = True
    except ImportError:
        SEQFUZZ_IMPORTED = False

class SeqFuzzAdapter(FuzzingStrategy):
    name = "seqfuzz"

    def __init__(self):
        self.siamese_model = None
        self.bench_noCrash = None

    def initialize(self, config):
        global SEQFUZZ_IMPORTED
        if SEQFUZZ_IMPORTED:
            try:
                import sys
                old_argv = sys.argv.copy()
                sys.argv = [old_argv[0]]
                self.siamese_model = predict_siamese.load_tapnet_mode()
                sys.argv = old_argv
                self.siamese_model.cpu()
                # Assuming the weights are located relative to seqfuzz/
                weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../seqfuzz/tapnet/data/weights/tapnet.pkl'))
                if os.path.exists(weights_path):
                    self.siamese_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                self.siamese_model.eval()
                
                self.bench_noCrash = Hyperparameter.bench_noCrash
                if len(self.bench_noCrash) == 0:
                    self.bench_noCrash = torch.zeros((1, Hyperparameter.Step, Hyperparameter.Dimension)).cpu()
                else:
                    self.bench_noCrash = torch.FloatTensor(np.array(self.bench_noCrash)).cpu()
                    if len(self.bench_noCrash.shape) == 2:
                        self.bench_noCrash = self.bench_noCrash.unsqueeze(0)
            except Exception as e:
                print(f"[SeqFuzz] Failed to load model: {e}")
                SEQFUZZ_IMPORTED = False

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
        pass

    def compute_feedback(self, candidate, result, features):
        scores = {}
        if SEQFUZZ_IMPORTED and self.siamese_model:
            obs_seq = result.get("obs_seq", [])
            # Only test if trajectory matches expected length
            if len(obs_seq) >= Hyperparameter.Step:
                # Use only the first Hyperparameter.Step
                input_seq = obs_seq[:Hyperparameter.Step]
                try:
                    # predict_once usually returns a hard class or distance
                    ret = predict_siamese.predict_once(self.siamese_model, self.bench_noCrash, input_seq)
                    scores["novelty_score"] = float(ret)
                except Exception:
                    scores["novelty_score"] = 0.0
        return scores
