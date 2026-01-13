import time
import gym
import numpy as np

import sys
from fuzz.executor import Executor
from sb3_contrib import TQC
from typing import Any, Tuple, List


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
        # 请根据实际路径调整
        return TQC.load(
            "D:\\code\\fuzzers\\BipedalWalker\\rl-trained-agents\\tqc\\BipedalWalkerHardcore-v3_1\\BipedalWalkerHardcore-v3.zip",
            device='cpu',
            custom_objects={"learning_rate":lambda _: 3e-4, "lr_schedule": lambda _: 3e-4}, 
            kwargs={'seed': 0, 'buffer_size': 1})


    def execute_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float, float, float, List]:
        '''
        Executes the model and returns the trajectory data and behaviour metrics.
        Returns: (acc_reward, is_crash, obs_seq, exec_time, bd_dist, bd_mean_angle, transitions)
        '''
        env = gym.make('BipedalWalkerHardcore-v3')
        env.seed(self.env_seed)
        obs_seq = []
        transitions = [] 
        
        acc_reward = 0.0

        obs = env.reset(input)
        state = None
        
        total_x_pos_sum = 0.0
        total_abs_angle_sum = 0.0
        episode_steps = 0
        
        t0 = time.time()
        for t in range(self.sim_steps):
            action, state = policy.predict(obs, state=state, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            
            # 记录 Transition
            transitions.append((obs.copy(), action[0].copy() if isinstance(action, np.ndarray) else action, reward, next_obs.copy(), done))

            real_env = env.unwrapped
            if hasattr(real_env, 'hull'):
                total_x_pos_sum += real_env.hull.position[0]
                total_abs_angle_sum += abs(real_env.hull.angle)
            
            episode_steps += 1
            acc_reward += reward
            obs_seq.append(obs)
            
            obs = next_obs
            
            if done:
                break

        env.close()
        
        bd_dist = total_x_pos_sum / max(1, episode_steps)
        bd_mean_angle = total_abs_angle_sum / max(1, episode_steps)

        return acc_reward, (reward == -100), np.array(obs_seq), time.time() - t0, bd_dist, bd_mean_angle, transitions


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    executor = BipedalWalkerExecutor(300, 0)
    input = executor.generate_input(rng)
    policy = executor.load_policy()
    reward, oracle, sequence, exec_time, bd_d, bd_a, trans = executor.execute_policy(input, policy)
    print(f"Input: {input}")
    print(f"Reward: {reward}, Crash: {oracle}, Time: {exec_time:.4f}")
    print(f"Transitions: {len(trans)}")