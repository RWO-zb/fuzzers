import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import DQN

# MountainCar-v0 state space bounds
MC_MIN_POS, MC_MAX_POS = -1.2, 0.6
MC_MIN_VEL, MC_MAX_VEL = -0.07, 0.07

def load_model(model_path):
    """Load a trained DQN model with zeroed-out training hyperparameters."""
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    return DQN.load(model_path, custom_objects=custom_objects)

def execute_policy(input_state, model, env_seed, descriptors=None, sim_steps=200):
    """
    Execute the policy from a given initial state and determine the oracle result.

    Returns:
        (total_reward, is_faulty, final_obs, final_obs, obs_sequence, empty_dict)
        Oracle: is_faulty=True if the car fails to reach position >= 0.5 within sim_steps.
    """
    env = gym.make('MountainCar-v0', render_mode=None)
    
    # Force the environment to start from the specified initial state
    obs, _ = env.reset(seed=env_seed)
    env.unwrapped.state = np.array(input_state, dtype=np.float32)
    obs = np.array(input_state, dtype=np.float32)
    
    total_reward = 0.0
    obs_seq = []
    
    for _ in range(sim_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        obs_seq.append(obs)
        
        if done or truncated:
            break
    
    env.close()
    
    final_pos = obs[0]
    # Oracle: failure to reach goal position (0.5) is considered a fault
    is_faulty = final_pos < 0.5
    
    return total_reward, is_faulty, obs, obs, obs_seq, {}

def get_edges(env_seed, descriptors, granularity=50):
    """Compute uniform grid edges for the MAP-Elites behavior space (position x velocity)."""
    x_edges = np.linspace(MC_MIN_POS, MC_MAX_POS, granularity + 1)
    y_edges = np.linspace(MC_MIN_VEL, MC_MAX_VEL, granularity + 1)
    return x_edges, y_edges

def compute_cell(behavior, xedges, yedges):
    """Map a behavior descriptor to its grid cell index."""
    x_idx = np.clip(np.searchsorted(xedges, behavior[0]) - 1, 0, len(xedges) - 2)
    y_idx = np.clip(np.searchsorted(yedges, behavior[1]) - 1, 0, len(yedges) - 2)
    return np.array([x_idx, y_idx])