import os

class Config:
    def __init__(self):
        self.env_id = "BipedalWalkerHardcore-v3"
        self.budget = 10000
        self.seed = 0
        self.alpha = 1.0 # crash-proneness weight
        self.beta = 1.0  # novelty weight
        self.gamma = 1.0 # diversity weight
        self.delta = 1.0 # uncertainty/curiosity weight
        self.eta = 1.0   # g-model/generative-model guidance weight
        self.lambda_cost = 0.1 # execution cost penalty
        
        self.scheduler_type = "ucb" # epsilon_greedy or ucb
        self.output_dir = "hybridfuzz/results/"
        self.model_path = "rl-trained-agents/tqc/BipedalWalkerHardcore-v3_1/BipedalWalkerHardcore-v3.zip"
        self.reward_fault_threshold = 10.0
        self.qdfuzz_env_seed = 723
        self.qd_descriptors = [4, 8]
        self.max_pool_size = 1000
        self.uncertainty_norm = "rolling" # rolling or log
        self.bootstrap_budget = 10
        self.uncertainty_threshold = 0.5
        self.reward_drop_threshold = 0.0
        self.g_model_threshold = 0.5
        self.seq_novelty_threshold = 0.5
        self.g_model_method = "generative+novelty"
        self.g_model_train_step = 50
        self.g_model_grid = 5
        self.seq_cvg_threshold = 0.02
        self.reward_drop_scale = 20.0
        self.disable_shared_pool = False
        
        # RL Algo configuration
        self.algo = "tqc"
        self.n_timesteps = 300

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
                
config = Config()
