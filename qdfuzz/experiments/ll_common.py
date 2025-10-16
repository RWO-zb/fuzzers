import json
import sys
import os
import time
import torch
import tqdm
import pandas as pd
import numpy as np

from typing import List, Tuple
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO
import gym

'''
Lunar Lander problem use case study.
Here, the input space is 2d, and it only describes the initial force applied to the lander.
'''

############################ CONSTANTS ################################


# default values
DEFAULT_MIN = -1000
DEFAULT_MAX = 1000
DEFAULT_MIN_INPUT = np.array([DEFAULT_MIN, DEFAULT_MIN])
DEFAULT_MAX_INPUT = np.array([DEFAULT_MAX, DEFAULT_MIN])
DEFAULT_MAX_DIST_INPUT: np.ndarray = np.linalg.norm(DEFAULT_MAX_INPUT - DEFAULT_MIN_INPUT)


###################### EXECUTION/EXPERIMENT SUPPORTERS ################################


def generate_input(rng: np.random.Generator, lows: List[float], highs: List[float]):
    '''Generates a single input between the given bounds (parameters).'''
    return rng.uniform(low=lows, high=highs, size=2)


def generate_inputs(rng: np.random.Generator, lows: List[float], highs: List[float], n: int):
    '''Generates @n inputs with the lower and upper bounds parameters.'''
    return rng.uniform(low=lows, high=highs, size=(n, 2))


def load_lunar_lander_model():
    '''Loads the model under test.'''
    custom_objects = {
        'learning_rate': 0.0,
        'lr_schedule': lambda _: 0.0,
        'clip_range': lambda _: 0.0,
    }
    return PPO.load('rl-trained-agents/ppo/LunarLander-v2_1/LunarLander-v2.zip', custom_objects=custom_objects)


def execute_policy(input: np.ndarray, model: BaseAlgorithm, env_seed: int, sim_steps: int = 1000) -> Tuple[float, bool, np.ndarray, np.ndarray, float]:
    '''Executes the model on the environment and only computes the hand-coded behavior. It also returns the final state.'''
    t0 = time.time()
    env: gym.Env = gym.make('LunarLander-v3')
    env.seed(env_seed)
    obs = env.reset(input)
    state = None
    acc_reward = 0.0

    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []

    for _ in range(sim_steps):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, info = env.step(action)
        acc_reward += reward

        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])
        all_y_vels.append(y_vel)
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

        if done:
            break


    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)
    behavior = np.array([impact_x_pos, impact_y_vel])
    env.close()
    exec_time = time.time() - t0
    return acc_reward, (reward == -100), behavior, obs, exec_time


def execute_policy_trajectory(input: np.ndarray, model: BaseAlgorithm, env_seed: int, sim_steps: int = 1000) -> Tuple[float, bool, np.ndarray, np.ndarray, float]:
    '''
    Executes the model with the simulator and returns state sequence. Useful for MDPFuzz.
    '''
    t0 = time.time()
    env: gym.Env = gym.make('LunarLander-v3')
    env.seed(env_seed)
    obs = env.reset(input)
    state = None
    acc_reward = 0.0

    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []

    obs_seq = []

    for _ in range(sim_steps):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, info = env.step(action)
        acc_reward += reward

        obs_seq.append(obs)

        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])
        all_y_vels.append(y_vel)
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

        if done:
            break


    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)
    behavior = np.array([impact_x_pos, impact_y_vel])
    env.close()
    exec_time = time.time() - t0

    return acc_reward, (reward == -100), behavior, np.array(obs_seq), exec_time


def get_edges(env_seed: int, sim_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(f'grid/ll/{env_seed}_{sim_steps}_xedges.npy'), np.load(f'grid/ll/{env_seed}_{sim_steps}_yedges.npy')


if __name__ == '__main__':
    torch.set_num_threads(1)
    main_seed = 2021
    env_seed = 0
    model = load_lunar_lander_model()

    rng: np.random.Generator = np.random.default_rng(main_seed)

    oracles, rewards, behaviors, final_states = [], [], [], []
    inputs = []
    for _ in tqdm.tqdm(range(100)):
        input: np.ndarray = rng.uniform(low=DEFAULT_MIN, high=DEFAULT_MAX, size=2)
        r, o, b, fs, _ = execute_policy(input, model, env_seed, 1000)
        oracles.append(o)
        rewards.append(r)
        behaviors.append(b)
        final_states.append(fs)

        inputs.append(input)

    np.save('behaviors.npy',np.vstack(behaviors))
    np.save('inputs.npy', np.vstack(inputs))
    import matplotlib.pyplot as plt
    print('test OK')
    fig, ax = plt.subplots()
    behaviors = np.array(behaviors)
    ax.scatter(behaviors[:, 0], behaviors[:, 1], s=10, alpha=0.5)
    fig.savefig('ll_test.png')