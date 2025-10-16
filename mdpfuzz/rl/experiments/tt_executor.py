import time
import gym
import numpy as np

from map_builder import MapBuilder
from taxi_agent import TestAgent
from typing import Any, Tuple

import sys
sys.path.append('../../methods/src/')
from executor import Executor


POLICY_FILEPATH = 'taxi_large_map_qtable_1299999.npy'
MAP_FILEPATH = 'map_large.txt'
INPUT_LOWS = [0, 0, 0, 0]
INPUT_UPS = [18, 13, 11, 11]
PASS_IN_TAXI_IDX = 11


class TaxiExecutor(Executor):

    def __init__(self, sim_steps, env_seed, map_fp: str = MAP_FILEPATH) -> None:
        super().__init__(sim_steps, env_seed)
        self._map = MapBuilder(map_fp)
        self._env = gym.make('Taxi-v3', map=self._map.map)


    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        input = rng.integers(low=INPUT_LOWS, high=INPUT_UPS, size=4)
        # checks if the passenger is already at the destination
        if  input[2] == input[3]:
            return self.generate_input(rng)
        else:
            return input


    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        inputs = []
        while len(inputs) < n:
            inputs.append(self.generate_input(rng))
        return np.array(inputs, dtype=int)


    def mutate(self, input: np.ndarray, rng: np.random.Generator, idx: int = None, **kwargs) -> np.ndarray:
        mutant = input.copy()
        if idx is None:
            idx = rng.integers(0, 4)
        tmp = np.arange(INPUT_UPS[idx]).tolist()
        value = mutant[idx]

        weights = 1 / (np.abs(tmp - value) + 1)
        weights[value] = 0
        probs = weights / np.sum(weights)
        mutant[idx] = rng.choice(a=tmp, size=1, p=probs)

        # the passenger location and its destination must be different
        if (idx == 2) and (mutant[idx] == mutant[idx + 1]):
            return self.mutate(input, rng, idx)
        elif (idx == 3) and (mutant[idx - 1] == mutant[idx]):
            return self.mutate(input, rng, idx)
        else:
            return mutant


    def load_policy(self, fp: str = POLICY_FILEPATH):
        return TestAgent(fp)


    def execute_policy(self, input: np.ndarray, policy: Any) -> Tuple[float, bool, np.ndarray, float]:
        '''Executes the model and returns the trajectory data. Useful for MDPFuzz.'''
        t0 = time.time()
        obs = self._env.reset(input)
        acc_reward = 0.0
        done = False
        oracle = False

        obs_seq = [list(self._env.decode(obs))]

        while not done:
            action = policy.step(obs)
            obs, reward, done, info = self._env.step(action)
            acc_reward += reward
            obs_seq.append(list(self._env.decode(obs)))

            if not oracle:
                oracle = (reward == -10) or (info.get('crash', False))

            if done or oracle:
                break
        # error when adding dtype=int...
        return acc_reward, oracle, np.vstack(obs_seq), time.time() - t0


if __name__ == '__main__':
    rng = np.random.default_rng()
    executor: Executor = TaxiExecutor(0, 0) # no timestep limit
    input = executor.generate_input(rng)
    policy = executor.load_policy()
    reward, oracle, sequence, exec_time = executor.execute_policy(input, policy)
    print(input, reward, oracle, exec_time)
    print(sequence.shape, sequence)