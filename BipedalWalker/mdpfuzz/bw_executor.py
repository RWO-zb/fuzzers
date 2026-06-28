import time
import gym
import numpy as np
import torch  # [新增] 用于张量计算
import pickle
from pathlib import Path

import sys
from fuzz.executor import Executor
from sb3_contrib import TQC
from stable_baselines3 import PPO
from typing import Any, Tuple, List


class BipedalWalkerExecutor(Executor):
    def __init__(self, sim_steps, env_seed, save_physics=False):
        """
        初始化 Executor
        """
        super().__init__(sim_steps, env_seed)
        self.save_physics = save_physics
        self.crash_physics_trajectories = []
        self.policy_vecnormalizers = {}

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

    def load_policy(self, algo: str = "tqc", model_path: str = None, vecnormalize_path: str = None):
        project_root = Path(__file__).resolve().parents[1]
        algo = algo.lower()
        default_paths = {
            "tqc": project_root / "rl-trained-agents" / "tqc" / "BipedalWalkerHardcore-v3_1" / "BipedalWalkerHardcore-v3.zip",
            "ppo": project_root / "rl-trained-agents" / "ppo" / "BipedalWalkerHardcore-v3_1" / "BipedalWalkerHardcore-v3.zip",
        }
        model_classes = {
            "tqc": TQC,
            "ppo": PPO,
        }
        if algo not in model_classes:
            raise ValueError(f"Unsupported RL algorithm: {algo}. Supported: {sorted(model_classes)}")

        if model_path is None:
            model_path = default_paths[algo]

        custom_objects = {
            "learning_rate": lambda _: 3e-4,
            "lr_schedule": lambda _: 3e-4,
            "clip_range": lambda _: 0.2,
        }
        load_kwargs = {
            "device": "cpu",
            "custom_objects": custom_objects,
        }
        if algo == "tqc":
            load_kwargs["kwargs"] = {"seed": 0, "buffer_size": 1}

        policy = model_classes[algo].load(str(model_path), **load_kwargs)

        vecnormalize = None
        if vecnormalize_path is None and algo == "ppo":
            candidate = Path(model_path).with_name("BipedalWalkerHardcore-v3") / "vecnormalize.pkl"
            if candidate.exists():
                vecnormalize_path = candidate
        if vecnormalize_path:
            with open(vecnormalize_path, "rb") as f:
                vecnormalize = pickle.load(f)
            vecnormalize.training = False

        self.policy_vecnormalizers[id(policy)] = vecnormalize

        return policy

    def normalize_observation(self, obs: np.ndarray, policy: Any) -> np.ndarray:
        vecnormalize = self.policy_vecnormalizers.get(id(policy))
        if vecnormalize is None:
            return obs
        return vecnormalize.normalize_obs(obs)

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

    def execute_policy(
        self,
        input: np.ndarray,
        policy: Any,
        record_physics: bool = True,
    ) -> Tuple[float, bool, np.ndarray, float, List, bool]:
        '''
        执行策略并返回轨迹数据。
        '''
        # [修改] 根据论文定义，仿真/评估时间应包含环境初始化与重置开销
        t0 = time.time()
        
        env = gym.make('BipedalWalkerHardcore-v3')
        
        try:
            env.reset(seed=int(self.env_seed))
        except:
            env.seed(self.env_seed)
            
        transitions = []
        
        current_episode_physics = []
        acc_reward = 0.0

        try:
            obs = env.reset(input)
        except TypeError:
            obs = env.reset()
        obs_seq = [obs.copy()]
        
        state = None
        episode_steps = 0
        
        if self.save_physics and record_physics:
            phys = self.extract_physics_state(env)
            if phys:
                current_episode_physics.append(phys)
        
        # =============================================================================
        # --- 核心优化循环 ---
        # =============================================================================
        for t in range(self.sim_steps):
            # [优化] Fast Predict：绕过 SB3 繁重的 distribution 对象实例化
            model_obs = self.normalize_observation(obs, policy)
            obs_tensor = torch.as_tensor(model_obs).float().unsqueeze(0).to("cpu")
            with torch.no_grad():
                # 针对 TQC / SAC 算法
                if hasattr(policy, "policy") and hasattr(policy.policy, "actor"):
                    mean_actions, _, _ = policy.policy.actor.get_action_dist_params(obs_tensor)
                    action = torch.tanh(mean_actions).squeeze(0).cpu().numpy()
                else:
                    action, _ = policy.predict(model_obs, state=state, deterministic=True)
            
            next_obs, reward, done, info = env.step(action)
            
            if self.save_physics and record_physics:
                phys = self.extract_physics_state(env)
                if phys:
                    current_episode_physics.append(phys)

            if isinstance(action, np.ndarray):
                if action.ndim == 2: 
                    act_save = action[0].copy()
                else: 
                    act_save = action.copy()
            else:
                act_save = action
            
            transitions.append((obs.copy(), act_save, reward, next_obs.copy(), done))
            
            episode_steps += 1
            acc_reward += reward
            obs_seq.append(next_obs.copy())
            
            obs = next_obs
            
            if done:
                break

        env.close()
        
        # 1. 判定物理崩溃 (躯干触地)：读取底层 Box2D 物理引擎状态
        is_physical_crash = bool(getattr(env.unwrapped, 'game_over', False))
        
        # 2. 判定性能/奖励故障：没有发生物理跌倒，但是总得分过低
        did_crash = bool(is_physical_crash)
        is_reward_fault = bool((acc_reward < 10) and not is_physical_crash)
        
        # 总的 Crash 标志
        is_failure = bool(is_physical_crash or is_reward_fault)

        if is_failure and self.save_physics and record_physics:
            if len(current_episode_physics) > 20:
                self.crash_physics_trajectories.append({
                    "seed": input,
                    "trajectory": current_episode_physics,
                    "fault_type": "physical_crash" if did_crash else "reward_fault"
                })

        return acc_reward, is_failure, np.array(obs_seq), time.time() - t0, transitions, is_physical_crash
