import time
import gym
import numpy as np

import sys
sys.path.append('../../methods/src/')
from executor import Executor
from sb3_contrib import TQC
from typing import Any, Tuple


class BipedalWalkerExecutor(Executor):

    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(low=1, high=4, size=15)


    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.integers(low=1, high=4, size=(n, 15))


    def mutate(self, input: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        mutation = rng.choice(2, 15, p=[0.9, 0.1])
        if np.sum(mutation) == 0:
            mutation[0] = 1
        mutated_input = input + mutation
        mutated_input = np.remainder(mutated_input, 4)
        mutated_input = np.clip(mutated_input, 1, 3)
        return mutated_input


    def load_policy(self):
        return TQC.load(
            "D:/code/rl-trained-agents/tqc/BipedalWalker-v3_1/BipedalWalker-v3.zip",
            device='cpu',
            custom_objects={"learning_rate":lambda _: 3e-4, "lr_schedule": lambda _: 3e-4}, 
            kwargs={'seed': 0, 'buffer_size': 1})


    def execute_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float]:
        '''Executes the model and returns the trajectory data. Useful for MDPFuzz.'''
        env = gym.make('BipedalWalkerHardcore-v3')
        env.seed(self.env_seed)
        obs_seq = []
        acc_reward = 0.0

        obs = env.reset(input)
        state = None
        t0 = time.time()
        for t in range(self.sim_steps):
            action, state = policy.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            acc_reward += reward
            obs_seq.append(obs)
            if done:
                break

        env.close()
        return acc_reward, (reward == -100), np.array(obs_seq), time.time() - t0


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    executor: Executor = BipedalWalkerExecutor(300, 0)
    input = executor.generate_input(rng)
    policy = executor.load_policy()
    reward, oracle, sequence, exec_time = executor.execute_policy(input, policy)
    print(input, reward, oracle, exec_time)
    print(sequence.shape)