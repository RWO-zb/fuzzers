import os
import time
import torch
import tqdm
import numpy as np
from typing import List, Tuple, Iterable

from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib import TQC

import gym


'''
Bipedal Walker problem use case study.
'''


############################ CONSTANTS ################################

# Leo Cazenille's naming scheme
FEATURES = [
    'meanDistance',
    'meanHeadStability',
    'meanTorquePerStep',
    'meanJump',
    'meanLeg0HipAngle',
    'meanLeg0HipSpeed',
    'meanLeg0KneeAngle',
    'meanLeg0KneeSpeed',
    'meanLeg1HipAngle',
    'meanLeg1HipSpeed',
    'meanLeg1KneeAngle',
    'meanLeg1KneeSpeed'
]
# Input space according to MDPFuzz
MIN_INPUT = np.array([1 for _ in range(15)])
MAX_INPUT = np.array([3 for _ in range(15)])
MAX_DIST_INPUT: np.ndarray = np.linalg.norm(MAX_INPUT - MIN_INPUT)
AVG_SIZE = 30
EXPERT_INDICES = [
    [0, 1],
    [2, 3],
    [4, 8],
    [5, 11]
]
EXPERT_PLOT_ARGS = [
    {'xlabel': 'distance to the goal', 'ylabel': 'hull angle', 'title': 'Distance vs Hull angle'},
    {'xlabel': 'torque (actions)', 'ylabel': 'jump rate', 'title': 'Torque vs Jump'},
    {'xlabel': '1st leg', 'ylabel': '2nd leg', 'title': 'Hip angles'},
    {'xlabel': '1st leg', 'ylabel': '2nd leg', 'title': 'Hip speeds'}
]

###################### EXECUTION/EXPERIMENT SUPPORTERS ################################


def generate_input(rng: np.random.Generator = None):
    if rng is None:
        return np.random.randint(low=1, high=4, size=15)
    else:
        return rng.integers(low=1, high=4, size=15)


def generate_inputs(rng: np.random.Generator, n: int):
    return rng.integers(low=1, high=4, size=n)


def load_model():
    return TQC.load('rl-trained-agents/tqc/BipedalWalkerHardcore-v3_1/BipedalWalkerHardcore-v3.zip',
                    custom_objects={
                        "learning_rate": lambda _: 3e-4,
                        "lr_schedule": lambda _: 3e-4,
                     }, 
                    kwargs={'seed': 0, 'buffer_size': 1})


def get_key(input: np.ndarray):
    '''Integer like representation of the float numpy arrays as keys.'''
    return ' '.join([f'{i:.0f}' for i in input])


def get_input_from_key(key: str) -> np.ndarray:
    return np.asfarray(key.split(' '), dtype=str).astype(int)


def get_inputs_from_keys(keys: Iterable[str]) -> np.ndarray:
    return np.array([np.asfarray(k.split(' '), dtype=str).astype(int) for k in keys])


def execute_policy(input: np.ndarray, model: BaseAlgorithm, env_seed: int, descriptors: List = None, sim_steps: int = 300) -> Tuple[float, bool, np.ndarray, np.ndarray, float]:
    '''Executes the model on the environment and only computes the 12 features used by Leo Cazenille. It also returns the final state.'''

    env = gym.make('BipedalWalkerHardcore-v4',rand_seed=env_seed)

    acc_reward = 0.0
    features = np.zeros(12)

    obs = env.reset(input)
    state = None
    t0 = time.time()
    for t in range(sim_steps):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, info = env.step(action)
        features += info['features'] # numpy array
        acc_reward += reward

        if done:
            break

    env.close()
    features /= t
    exec_time = time.time() - t0

    if descriptors is not None:
        descriptors = np.array(descriptors)
        assert all(descriptors < 12) and all(descriptors >= 0)
        return acc_reward, (reward == -100), features[descriptors], obs, exec_time
    else:
        return acc_reward, (reward == -100), features, obs, exec_time


def execute_policy_trajectory(input: np.ndarray, model: BaseAlgorithm, env_seed: int, sim_steps: int = 300) -> Tuple[float, bool, np.ndarray, List[np.ndarray], float]:
    '''Executes the model and returns the trajectory data. Useful for MDPFuzz.'''
    env = gym.make('BipedalWalkerHardcore-v3', rand_seed=env_seed)
    features = np.zeros(12)
    obs_seq = []
    acc_reward = 0.0

    obs = env.reset(input)
    state = None
    t0 = time.time()
    for t in range(sim_steps):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, info = env.step(action)
        features += info['features'] # numpy array
        acc_reward += reward
        obs_seq.append(obs)
        if done:
            break

    env.close()
    features /= t
    exec_time = time.time() - t0
    return acc_reward, (reward == -100), features, np.array(obs_seq), exec_time


def get_edges(env_seed: int, descriptors: np.ndarray, sim_steps: int = 300) -> np.ndarray:
    '''Returns the saved grid edges.'''
    edges = np.load(f'grid/bw/{env_seed}_{sim_steps}_edges.npy')
    return edges[descriptors]


if __name__ == '__main__':
    torch.set_num_threads(1)
    main_seed = 2021
    env_seed = 0
    model = load_model()

    rng: np.random.Generator = np.random.default_rng(main_seed)
    descriptors = EXPERT_INDICES[0]
    oracles, rewards, behaviors, final_states = [], [], [], []

    for _ in tqdm.tqdm(range(100)):
        input: np.ndarray = rng.integers(low=1, high=4, size=15)
        r, o, b, fs, _ = execute_policy(input, model, env_seed, descriptors, 300)
        oracles.append(o)
        rewards.append(r)
        behaviors.append(b)
        final_states.append(fs)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    behaviors = np.array(behaviors)
    print(behaviors.shape)
    ax.scatter(behaviors[:, 0], behaviors[:, 1], s=10, alpha=0.5)
    fig.savefig('bw_test.png')