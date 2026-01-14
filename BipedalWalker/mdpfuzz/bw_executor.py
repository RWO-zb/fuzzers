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
        :param save_physics: 是否开启物理轨迹收集 (用于反向课程生成)
        """
        super().__init__(sim_steps, env_seed)
        self.save_physics = save_physics
        # 用于存储导致 Crash 的物理轨迹: [{'seed': input, 'trajectory': [state_t0, state_t1, ...]}, ...]
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
        # 请根据实际路径调整
        return TQC.load(
            "D:\\code\\fuzzers\\BipedalWalker\\rl-trained-agents\\tqc\\BipedalWalkerHardcore-v3_1\\BipedalWalkerHardcore-v3.zip",
            device='cpu',
            custom_objects={"learning_rate":lambda _: 3e-4, "lr_schedule": lambda _: 3e-4}, 
            kwargs={'seed': 0, 'buffer_size': 1})

    def extract_physics_state(self, env):
        """
        从底层 Box2D 环境提取物理状态
        """
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
        Returns: (acc_reward, is_crash, obs_seq, exec_time, bd_dist, bd_mean_angle, transitions)
        '''
        env = gym.make('BipedalWalkerHardcore-v3')
        # input 即为 seed 数组 (mutate_states)
        try:
            env.reset(seed=int(0)) # 初始化随机种子
        except:
            env.seed(0)
            
        obs_seq = []
        transitions = [] 
        
        # 当前 Episode 的物理轨迹缓存
        current_episode_physics = []
        
        acc_reward = 0.0

        # 注意：这里 reset 传入 input，这是自定义环境接收地形参数的方式
        try:
            obs = env.reset(input)
        except TypeError:
            # 兼容性处理
            obs = env.reset()
        
        state = None
        
        total_x_pos_sum = 0.0
        total_abs_angle_sum = 0.0
        episode_steps = 0
        
        # 记录初始物理状态
        if self.save_physics:
            phys = self.extract_physics_state(env)
            if phys:
                current_episode_physics.append(phys)
        
        t0 = time.time()
        # 确保 reward 在循环外有定义，防止 sim_steps 为 0 或直接 break 的极端情况
        reward = 0.0 
        
        for t in range(self.sim_steps):
            action, state = policy.predict(obs, state=state, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            
            # 记录每一步的物理状态
            if self.save_physics:
                phys = self.extract_physics_state(env)
                if phys:
                    current_episode_physics.append(phys)

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
        
        # [修改点] 判定逻辑修改：摔倒(-100) 或者 表现极差(总分<10) 都视为 Crash
        is_crash = (reward == -100) or (acc_reward < 10)

        # 如果发生 Crash 且开启了物理保存，则存储该轨迹
        if is_crash and self.save_physics:
            # 只有当轨迹长度足够时才保存（可选，防止保存开局即死的无效数据）
            if len(current_episode_physics) > 20:
                self.crash_physics_trajectories.append({
                    "seed": input,  # 保存当前的地形参数 input
                    "trajectory": current_episode_physics
                })

        return acc_reward, is_crash, np.array(obs_seq), time.time() - t0, bd_dist, bd_mean_angle, transitions


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    # 测试 save_physics 功能
    executor = BipedalWalkerExecutor(300, 0, save_physics=True)
    input = executor.generate_input(rng)
    policy = executor.load_policy()
    reward, oracle, sequence, exec_time, bd_d, bd_a, trans = executor.execute_policy(input, policy)
    print(f"Input: {input}")
    print(f"Reward: {reward}, Crash: {oracle}, Time: {exec_time:.4f}")
    print(f"Transitions: {len(trans)}")
    print(f"Physics Trajectories Saved: {len(executor.crash_physics_trajectories)}")