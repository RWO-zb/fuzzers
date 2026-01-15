import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import DQN

# MountainCar-v0 状态范围
MC_MIN_POS, MC_MAX_POS = -1.2, 0.6
MC_MIN_VEL, MC_MAX_VEL = -0.07, 0.07

def load_model(model_path):
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    return DQN.load(model_path, custom_objects=custom_objects)

def execute_policy(input_state, model, env_seed, descriptors=None, sim_steps=200):
    env = gym.make('MountainCar-v0', render_mode=None)
    
    # 强制设置初始状态
    obs, _ = env.reset(seed=env_seed)
    env.unwrapped.state = np.array(input_state, dtype=np.float32)
    obs = np.array(input_state, dtype=np.float32)
    
    total_reward = 0.0
    
    # [修改] 用于存储轨迹 (s_1, s_2, ...)
    obs_seq = []
    
    for _ in range(sim_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        # [修改] 记录 Step 之后的观测值
        obs_seq.append(obs)
        
        if done or truncated:
            break
    
    env.close()
    
    final_pos = obs[0]
    # Oracle: 没到达 0.5 视为 Fault (True)
    is_faulty = final_pos < 0.5
    
    # [修改] 返回值包含 obs_seq
    return total_reward, is_faulty, obs, obs, obs_seq, {}

def get_edges(env_seed, descriptors, granularity=50):
    x_edges = np.linspace(MC_MIN_POS, MC_MAX_POS, granularity + 1)
    y_edges = np.linspace(MC_MIN_VEL, MC_MAX_VEL, granularity + 1)
    return x_edges, y_edges

def compute_cell(behavior, xedges, yedges):
    x_idx = np.clip(np.searchsorted(xedges, behavior[0]) - 1, 0, len(xedges) - 2)
    y_idx = np.clip(np.searchsorted(yedges, behavior[1]) - 1, 0, len(yedges) - 2)
    return np.array([x_idx, y_idx])