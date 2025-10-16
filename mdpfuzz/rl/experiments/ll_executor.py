import time
import gym
import numpy as np

import sys
sys.path.append('../../methods/src/')
from executor import Executor
from typing import Any, Tuple, List
from stable_baselines3 import PPO


DEFAULT_MIN = -1000
DEFAULT_MAX = 1000
MUTATION_INTENSITY = 5.0


class LunarLanderExecutor(Executor):

    def generate_input(self, rng: np.random.Generator):
        '''Generates a single input between the given bounds (parameters).'''
        return rng.uniform(low=DEFAULT_MIN, high=DEFAULT_MAX, size=2)


    def generate_inputs(self, rng: np.random.Generator, n: int = 1):
        '''Generates @n inputs with the lower and upper bounds parameters.'''
        if n == 1:
            return self.generate_input(rng)
        else:
            return rng.uniform(low=DEFAULT_MIN, high=DEFAULT_MAX, size=(n, 2))


    def mutate(self, input: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        return np.clip(rng.normal(input, MUTATION_INTENSITY), [DEFAULT_MIN, DEFAULT_MIN], [DEFAULT_MAX, DEFAULT_MAX])


    def load_policy(self):
        custom_objects = {
        'learning_rate': 0.0,
        'lr_schedule': lambda _: 0.0,
        'clip_range': lambda _: 0.0,
        }
        return PPO.load('rl-trained-agents/ppo/LunarLander-v2_1/LunarLander-v2.zip', device='cpu', custom_objects=custom_objects)


    def execute_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float]:
        '''Executes the model and returns the trajectory data. Useful for MDPFuzz.'''
        t0 = time.time()
        env: gym.Env = gym.make('LunarLander-v3')
        env.seed(self.env_seed)
        obs = env.reset(input)
        state = None
        acc_reward = 0.0

        obs_seq = []

        for _ in range(self.sim_steps):
            action, state = policy.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            acc_reward += reward

            obs_seq.append(obs)

            if done:
                break

        env.close()
        return acc_reward, (reward == -100), np.array(obs_seq), time.time() - t0


# exec(open('ll_executor.py').read())
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    executor: Executor = LunarLanderExecutor(1000, 0)
    input = executor.generate_input(rng)
    policy = executor.load_policy()
    reward, oracle, sequence, exec_time = executor.execute_policy(input, policy)
    print(input, reward, oracle, exec_time)
    print(sequence.shape)