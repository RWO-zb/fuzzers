import time
import os
import gym
import tqdm
import numpy as np

from typing import Tuple, List
from map_builder import MapBuilder


'''
Taxi environment use-case.
Faults occur in case of illegal action or crash.
'''


############################ CONSTANTS ################################


POLICY_FILEPATH = 'taxi_large_map_qtable.npy'
MAP_FILEPATH = 'map_large.txt'
INPUT_LOWS = [0, 0, 0, 0]
INPUT_UPS = [18, 13, 11, 11]
PASS_IN_TAXI_IDX = 11


###################### EXECUTION/EXPERIMENT SUPPORTERS ################################


class TestAgent():
    '''Shallow class that loads a qtable and steps.'''
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Input file not found.')
        self.qtable = np.load(filepath)

    def step(self, state):
        return np.argmax(self.qtable[state])


def load_taxi_model(fp: str = POLICY_FILEPATH) -> TestAgent:
    '''Loads the model under test.'''
    return TestAgent(fp)


def generate_input(rng: np.random.Generator) -> np.ndarray:
    '''
    Generates a random input.
    Note that recursivity happens if the latter is malformed.
    '''
    input = rng.integers(low=INPUT_LOWS, high=INPUT_UPS, size=4)
    # checks if the passenger is already at the destination
    if  input[2] == input[3]:
        return generate_input(rng)
    else:
        return input


def generate_inputs(rng: np.random.Generator, n: int) -> np.ndarray:
    '''Generates @n inputs with the lower and upper bounds parameters.'''
    inputs = []
    while len(inputs) < n:
        inputs.append(generate_input(rng))
    return np.array(inputs, dtype=int)


# def generate_input_space() -> np.ndarray:
#     inputs = [
#         np.array([i, j, k, l])
#         for i in range(INPUT_LOWS[0], INPUT_UPS[0])
#         for j in range(INPUT_LOWS[1], INPUT_UPS[1])
#         for k in range(INPUT_LOWS[2], INPUT_UPS[2])
#         for l in range(INPUT_LOWS[3], INPUT_UPS[3])
#         if k != l]
#     return np.array(inputs, dtype=int)


def mutate(input: np.ndarray, rng: np.random.Generator = np.random.default_rng(), idx: int = None):
    mutant = input.copy()
    if idx is None:
        idx = rng.integers(0, 4)
    tmp = np.arange(INPUT_UPS[idx])
    value = mutant[idx]

    weights = np.abs(tmp - value)
    inversed_weights = np.max(weights) - weights
    inversed_weights[value] = 0.0
    probs = inversed_weights / sum(inversed_weights)
    mutant[idx] = rng.choice(tmp, p=probs)

    # the passenger location and its destination must be different
    if (idx == 2) and (mutant[idx] == mutant[idx + 1]):
        return mutate(input, rng)
    elif (idx == 3) and (mutant[idx - 1] == mutant[idx]):
        return mutate(input, rng)
    else:
        return mutant


def execute_policy(model: TestAgent, input: np.ndarray, env: gym.Env = gym.make('Taxi-v3')) -> Tuple[float, bool, np.ndarray, int, float]:
    '''
    Executes the model on the environment.
    It returns the final state and the hand-coded behavior.
    '''
    t0 = time.time()
    obs = env.reset(input)
    acc_reward = 0.0
    behavior = np.zeros(8)
    done = False
    oracle = False
    while not done:
        action = model.step(obs)
        obs, reward, done, info = env.step(action)
        acc_reward += reward

        # checks whether the passenger is in the taxi
        pass_in_taxi = list(env.decode(obs))[2] == PASS_IN_TAXI_IDX

        if action < 4:
            behavior[action + 4 * int(pass_in_taxi)] += 1

        if not oracle:
            oracle = (reward == -10) or (info.get('crash', False))

        if done or oracle:
            break

    exec_time = time.time() - t0
    return acc_reward, oracle, behavior, obs, exec_time


def execute_policy_trajectory(model: TestAgent, input: np.ndarray, env: gym.Env = gym.make('Taxi-v3')) -> Tuple[float, bool, np.ndarray, np.ndarray, float]:
    '''
    Executes the model with the simulator and returns the state sequence. Useful for MDPFuzz.
    '''
    t0 = time.time()
    obs = env.reset(input)
    acc_reward = 0.0
    behavior = np.zeros(8)
    done = False
    oracle = False
    obs_seq = [obs]
    while not done:
        action = model.step(obs)
        obs, reward, done, info = env.step(action)
        acc_reward += reward
        obs_seq.append(obs)

        pass_in_taxi = list(env.decode(obs))[2] == PASS_IN_TAXI_IDX

        if action < 4:
            behavior[action + 4 * int(pass_in_taxi)] += 1

        if not oracle:
            oracle = (reward == -10) or (info.get('crash', False))

        if done or oracle:
            break

    exec_time = time.time() - t0
    return acc_reward, oracle, behavior, np.array(obs_seq, dtype=int), exec_time


def get_taxi_env(map_fp: str = MAP_FILEPATH):
    map = MapBuilder(map_fp)
    return gym.make('Taxi-v3', map=map.map)


###################### BEHAVIOR SPACE ################################


def get_edges() -> Tuple[np.ndarray, np.ndarray]:
    fp = 'grid/tt/'
    lfp = fp + 'mins.npy'
    ufp = fp + 'maxs.npy'
    assert os.path.exists(lfp)
    assert os.path.exists(ufp)
    return np.load(lfp), np.load(ufp)


class BehaviorSpace():
    '''
    Implementation of a descriptor for the Taxi use-case.
    It computes 2d behaviors as the sum of the first and second half of the feature values, respectively.
    The container is a regular grid which evenly ranges from the minima and the maxima of the behaviors.
    '''

    def __init__(self, lower_bounds: np.ndarray = None, upper_bounds: np.ndarray = None) -> None:
        if (lower_bounds is None) or (upper_bounds is None):
            lower_bounds, upper_bounds = get_edges()
        n = len(lower_bounds)
        assert len(upper_bounds) == n
        tmp = int(n / 2)
        self.mins: np.ndarray = np.array([sum(lower_bounds[:tmp]), sum(lower_bounds[tmp:])], dtype=int)
        self.maxs: np.ndarray = np.array([sum(upper_bounds[:tmp]), sum(upper_bounds[tmp:])], dtype=int)
        self.x: List[int] = np.arange(self.mins[0], self.maxs[0] + 1).tolist()
        self.y: List[int] = np.arange(self.mins[1], self.maxs[1] + 1).tolist()


    def compute_behavior(self, feature: np.ndarray) -> np.ndarray:
        tmp = int(len(feature) / 2)
        return np.array([sum(feature[:tmp]), sum(feature[tmp:])], dtype=int)


    def compute_cell(self, behavior: np.ndarray) -> List[int]:
        assert len(behavior) == 2
        return [self.x.index(behavior[0]), self.y.index(behavior[1])]


    def describe(self, feature: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        '''Convenient function that returns the behavior and its cell of @feature.'''
        behavior = self.compute_behavior(feature)
        return behavior, self.compute_cell(behavior)


    def get_container(self) -> List[List[int]]:
        return [[i, j] for i in self.x for j in self.y]


######################################### MAIN ######################################################


def eval_map(model: TestAgent, map_fp: str = 'map_large.txt') -> Tuple[List[int], np.ndarray]:
    '''Evaluates the policy over the entire input space.'''
    map = MapBuilder(map_fp)
    env = gym.make('Taxi-v3', map=map.map)
    input_space = map.generate_input_space()
    evals = []
    n = len(input_space)
    for i in tqdm.tqdm(range(n)):
        input = input_space[i]
        obs = env.reset(input)
        failure = False
        done = False
        while not done:
            action = model.step(obs)
            obs, reward, done, info = env.step(action)
            if (reward == -10) or (info['crash'] == True):
                failure = True
                done = True
        evals.append(int(failure))
    env.close()
    return evals, input_space


# exec(open('tt_common.py').read())
if __name__ == '__main__':
    model = load_taxi_model()

    evals, inputs = eval_map(model)
    print(f'{(100 * sum(evals) / len(evals)):0.2f}% failure rate over the input space.')