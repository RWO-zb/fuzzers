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
        
        # RL Algo configuration
        self.algo = "tqc"
        self.n_timesteps = 300

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
                
config = Config()
