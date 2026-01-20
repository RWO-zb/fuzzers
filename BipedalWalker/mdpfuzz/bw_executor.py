import time
import gym
import numpy as np

import sys
from fuzz.executor import Executor
from sb3_contrib import TQC
from typing import Any, Tuple, List


class BipedalWalkerExecutor(Executor):
    def __init__(self, sim_steps, env_seed, save_physics=False):
        """
        初始化 Executor
        """
        super().__init__(sim_steps, env_seed)
        self.save_physics = save_physics
        self.crash_physics_trajectories = []

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
        # 请根据你的实际路径调整 path
        return TQC.load(
            "D:\\code\\fuzzers\\BipedalWalker\\rl-trained-agents\\tqc\\BipedalWalkerHardcore-v3_1\\BipedalWalkerHardcore-v3.zip",
            device='cpu',
            custom_objects={"learning_rate":lambda _: 3e-4, "lr_schedule": lambda _: 3e-4}, 
            kwargs={'seed': 0, 'buffer_size': 1})

    def extract_physics_state(self, env):
        base_env = env.unwrapped
        if not hasattr(base_env, 'hull') or not hasattr(base_env, 'legs'):
            return None

        hull = base_env.hull
        legs = base_env.legs 
        
        state_dict = {
            "hull_pos": (hull.position[0], hull.position[1]),
            "hull_angle": hull.angle,
            "hull_lin_vel": (hull.linearVelocity[0], hull.linearVelocity[1]),
            "hull_ang_vel": hull.angularVelocity,
            "legs": []
        }
        
        for leg_part in legs:
            leg_data = {
                "pos": (leg_part.position[0], leg_part.position[1]),
                "angle": leg_part.angle,
                "lin_vel": (leg_part.linearVelocity[0], leg_part.linearVelocity[1]),
                "ang_vel": leg_part.angularVelocity,
            }
            state_dict["legs"].append(leg_data)
            
        return state_dict

    def execute_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float, float, float, List]:
        '''
        Executes the model and returns the trajectory data and behaviour metrics.
        [Alignment]: 该函数返回的 obs 是原始环境 (gym.make) 生成的，因此天然是 RAW 数据。
        '''
        env = gym.make('BipedalWalkerHardcore-v3')
        
        try:
            env.reset(seed=int(0)) 
        except:
            env.seed(0)
            
        obs_seq = []
        transitions = [] # [Raw] Transitions
        
        current_episode_physics = []
        acc_reward = 0.0

        try:
            obs = env.reset(input)
        except TypeError:
            obs = env.reset()
        
        state = None
        
        total_x_pos_sum = 0.0
        total_abs_angle_sum = 0.0
        episode_steps = 0
        
        if self.save_physics:
            phys = self.extract_physics_state(env)
            if phys:
                current_episode_physics.append(phys)
        
        t0 = time.time()
        
        for t in range(self.sim_steps):
            # policy.predict 接受 Raw obs (前提是 Policy 训练时也见过 Raw，或者内部处理)
            # CureFuzz 中 Policy 接受归一化，但这里是 MDPFuzz 的执行器，
            # 只要这里保存出来的 transitions 中的 obs 是 Raw 的即可满足你的要求。
            # gym.make 生成的 env 没有 VecNormalize，所以 obs 是 Raw。
            action, state = policy.predict(obs, state=state, deterministic=True)
            
            next_obs, reward, done, info = env.step(action)
            
            if self.save_physics:
                phys = self.extract_physics_state(env)
                if phys:
                    current_episode_physics.append(phys)

            # [Alignment] 构造标准 5元组 (Raw Obs, Action, Reward, Raw Next Obs, Done)
            # 确保 action 是标量或数组的一致性处理
            act_save = action[0].copy() if isinstance(action, np.ndarray) else action
            transitions.append((obs.copy(), act_save, reward, next_obs.copy(), done))

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
        
        is_crash = (reward == -100) or (acc_reward < 10)

        if is_crash and self.save_physics:
            if len(current_episode_physics) > 20:
                self.crash_physics_trajectories.append({
                    "seed": input,
                    "trajectory": current_episode_physics
                })

        return acc_reward, is_crash, np.array(obs_seq), time.time() - t0, bd_dist, bd_mean_angle, transitions