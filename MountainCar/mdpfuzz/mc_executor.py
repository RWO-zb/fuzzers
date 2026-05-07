import sys
import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from fuzz.executor import Executor
from typing import Any, Tuple

class MountainCarExecutor(Executor):
    def __init__(self, sim_steps: int, env_seed: int = 0, model_path: str = ""):
        super().__init__(sim_steps, env_seed)
        self.model_path = model_path
        # MountainCar state bounds
        self.state_low = np.array([-0.6, 0], dtype=np.float32)
        self.state_high = np.array([-0.4, 0], dtype=np.float32)

    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        # Initial state bounds: pos [-0.6, -0.4], vel [0]
        pos = rng.uniform(-0.6, -0.4)
        vel = 0.0
        return np.array([pos, vel], dtype=np.float32)

    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        inputs = []
        for _ in range(n):
            inputs.append(self.generate_input(rng))
        return np.array(inputs)

    def mutate(self, input_state: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        # Gaussian mutation
        mutation_noise = rng.normal(0, 0.05, size=input_state.shape)
        mutated_state = input_state + mutation_noise
        mutated_state = np.clip(mutated_state, self.state_low, self.state_high)
        return mutated_state.astype(np.float32)

    def load_policy(self):
        # Load DQN model
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        return DQN.load(self.model_path, custom_objects=custom_objects)

    def execute_policy(self, input_state: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float]:
        # Create environment
        env = gym.make('MountainCar-v0', render_mode=None)
        
        obs_seq = []
        acc_reward = 0.0
        
        # Force set initial state
        obs, _ = env.reset(seed=self.env_seed)
        env.unwrapped.state = input_state
        obs = np.array(input_state, dtype=np.float32)
        
        state = None
        
        t0 = time.time()
        
        for t in range(self.sim_steps):
            action, state = policy.predict(obs, state=state, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            acc_reward += reward
            obs_seq.append(obs)
            
            if done or truncated:
                break
        
        env.close()
        
        # Oracle: Check if reached the goal (pos >= 0.5)
        # If final position is less than 0.5, consider it a crash (fuzzer target)
        final_pos = obs[0]
        is_crash = final_pos < 0.5
        
        exec_time = time.time() - t0
        
        return acc_reward, is_crash, np.array(obs_seq), exec_time

if __name__ == '__main__':
    # Simple self-test
    print("This is the executor module.")