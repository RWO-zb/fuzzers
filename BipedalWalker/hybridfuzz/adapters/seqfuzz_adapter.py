import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../seqfuzz')))

from hybridfuzz.strategy_base import FuzzingStrategy

try:
    import torch
    from seqfuzz.tapnet import predict_siamese, Hyperparameter
    from seqfuzz.fuzz.fuzz import fuzzing as SeqCoverageFuzzer
    SEQFUZZ_IMPORTED = True
except ImportError:
    try:
        import torch
        from tapnet import predict_siamese, Hyperparameter
        from fuzz.fuzz import fuzzing as SeqCoverageFuzzer
        SEQFUZZ_IMPORTED = True
    except ImportError:
        SEQFUZZ_IMPORTED = False

class SeqFuzzAdapter(FuzzingStrategy):
    name = "seqfuzz"

    def __init__(self):
        self.siamese_model = None
        self.bench_noCrash = None
        self.coverage_fuzzer = None
        self.imported = SEQFUZZ_IMPORTED
        self.model_loaded = False
        self.predict_success_count = 0
        self.predict_exception_count = 0
        self.predict_skipped_count = 0
        self.last_predict_value = None
        self.coverage_success_count = 0
        self.coverage_exception_count = 0
        self.last_coverage_value = None
        self.coverage_min = None
        self.coverage_max = None
        self.cvg_threshold = 0.02

    def _coverage_to_novelty(self, cvg):
        cvg = float(cvg)
        self.last_coverage_value = cvg
        if self.coverage_min is None:
            self.coverage_min = cvg
            self.coverage_max = cvg
            return 1.0
        self.coverage_min = min(self.coverage_min, cvg)
        self.coverage_max = max(self.coverage_max, cvg)
        denom = self.coverage_max - self.coverage_min
        if denom <= 1e-12:
            return 1.0
        # SeqFuzz treats low coverage/probability as interesting.
        return float(np.clip((self.coverage_max - cvg) / denom, 0.0, 1.0))

    def initialize(self, config):
        global SEQFUZZ_IMPORTED
        self.cvg_threshold = getattr(config, "seq_cvg_threshold", 0.02)
        if SEQFUZZ_IMPORTED:
            try:
                self.coverage_fuzzer = SeqCoverageFuzzer()
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
                self.model_loaded = True
                
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
                self.imported = False

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
        obs_seq = result.get("obs_seq", [])
        if self.coverage_fuzzer is not None and len(obs_seq) > 1:
            try:
                cvg = self.coverage_fuzzer.state_coverage(obs_seq)
                self.coverage_success_count += 1
                scores["raw_seq_coverage"] = float(cvg)
                scores["seq_below_threshold"] = bool(cvg < self.cvg_threshold)
                scores["novelty_score"] = self._coverage_to_novelty(cvg)
            except Exception:
                self.coverage_exception_count += 1
                scores["raw_seq_coverage"] = None
                scores["novelty_score"] = 0.0

        if SEQFUZZ_IMPORTED and self.siamese_model:
            # Only test if trajectory matches expected length
            if len(obs_seq) >= Hyperparameter.Step:
                # Use only the first Hyperparameter.Step
                input_seq = obs_seq[:Hyperparameter.Step]
                try:
                    # predict_once usually returns a hard class or distance
                    ret = predict_siamese.predict_once(self.siamese_model, self.bench_noCrash, input_seq)
                    self.last_predict_value = float(ret)
                    self.predict_success_count += 1
                    scores["seq_tapnet_prediction"] = self.last_predict_value
                except Exception:
                    scores["seq_tapnet_prediction"] = None
                    self.predict_exception_count += 1
            else:
                self.predict_skipped_count += 1
        else:
            self.predict_skipped_count += 1
        return scores

    def get_status(self):
        return {
            "imported": self.imported,
            "model_loaded": self.model_loaded,
            "predict_success_count": self.predict_success_count,
            "predict_exception_count": self.predict_exception_count,
            "predict_skipped_count": self.predict_skipped_count,
            "last_predict_value": self.last_predict_value,
            "coverage_success_count": self.coverage_success_count,
            "coverage_exception_count": self.coverage_exception_count,
            "last_coverage_value": self.last_coverage_value,
            "coverage_min": self.coverage_min,
            "coverage_max": self.coverage_max,
            "cvg_threshold": self.cvg_threshold,
        }
