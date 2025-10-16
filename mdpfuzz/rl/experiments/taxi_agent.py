import os
import random
import re
import tqdm
from abc import ABC, abstractmethod

import gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

INPUT_LOWS = [0, 0, 0, 0]
INPUT_UPS = [5, 5, 4, 4]

BIG_MAP = [
    "+-----------+",
    "| : : : : : |",
    "| : : : | : |",
    "|R: | : :G: |",
    "| : | : : : |",
    "| : : : | : |",
    "| : : : |W: |",
    "| : : : : : |",
    "| | : | : : |",
    "|Y| : |B: : |",
    "| : : : : : |",
    "+-----------+",
    ]

INPUT_LOWS_BIG = [0, 0, 0, 0]
INPUT_UPS_BIG = [10, 6, 5, 5]


from map_builder import MapBuilder


def generate_input_space(lows: List[int] = INPUT_LOWS, highs: List[int] = INPUT_UPS) -> np.ndarray:
    assert (len(lows) == 4) and (len(highs) == 4)
    inputs = [
        np.array([i, j, k, l])
        for i in range(lows[0], highs[0])
        for j in range(lows[1], highs[1])
        for k in range(lows[2], highs[2])
        for l in range(lows[3], highs[3])
        if k != l]
    return np.array(inputs, dtype=int)


def generate_big_input_space() -> np.ndarray:
    inputs = [
        np.array([i, j, k, l])
        for i in range(INPUT_LOWS_BIG[0], INPUT_UPS_BIG[0])
        for j in range(INPUT_LOWS_BIG[1], INPUT_UPS_BIG[1])
        for k in range(INPUT_LOWS_BIG[2], INPUT_UPS_BIG[2])
        for l in range(INPUT_LOWS_BIG[3], INPUT_UPS_BIG[3])
        if k != l]
    return np.array(inputs, dtype=int)


class Agent(ABC):
    '''
    Abstract class which represents an agent.
    An agent should be trained, used (to predicate next action), and imported / exported.
    '''

    @abstractmethod
    def train(self, nb_episodes: int, nb_episode_steps: int, result_filename='output'):
        '''
        Trains the agent to solve the MDP.
        @nb_episodes is the number of episodes.
        @nb_episode_steps is the maximum length of each episode.
        '''
        pass


    @abstractmethod
    def step(self, state):
        '''Returns the next action to do given @state.'''
        pass


    @abstractmethod
    def save(self, filename: str, erase=False):
        pass


    @abstractmethod
    def load(self, filename: str):
        pass


class TestAgent(Agent):
    '''Shallow class that loads a qtable and steps.'''
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Input file not found.')
        self.qtable = np.load(filepath)

    def step(self, state):
        return np.argmax(self.qtable[state])

    def load(self, filename: str):
        raise NotImplementedError()

    def save(self, filename: str, erase=False):
        raise NotImplementedError()

    def train(self, nb_episodes: int, nb_episode_steps: int, result_filename='output'):
        raise NotImplementedError()


class ExplicitAgent(Agent):
    '''
    Implementation of the abstract class Agent with an explicit Q-learning algorithm.
    Each state is thus associated with a probability distribution over the action space.
    It is therefore only suitable for discrete environments.
    '''

    def __init__(self, env: gym.wrappers.TimeLimit, lr: float, gamma: float, min_epsilon: float = 0.1):
        self._env = env
        self._nb_training_steps = 0
        self._lr = lr
        self._gamma = gamma
        self._min_epsilon = min_epsilon

        self._qtable = self._init_qtable()


    def _init_qtable(self):
        if self._nb_training_steps != 0:
            self._nb_training_steps = 0
            print('Agent re-initialized.')
        action_size = self._env.action_space.n
        state_size = self._env.observation_space.n
        return np.zeros((state_size, action_size))


    def train(self, nb_episodes: int, max_timesteps: int, **kwargs) -> List[float]:
        log_interval = kwargs.get('log_interval', None)

        exploration_ratio = kwargs.get('exploration_ratio', None)
        if exploration_ratio is not None:
            assert exploration_ratio > 0.0 and exploration_ratio < 1.0

        def compute_epsilon(i, imax: int, fraction: float):
            '''Linearly decreasing epsilon until @fraction times the training budget is reached.'''
            tmp = int(fraction * imax)
            epsilon = self._min_epsilon
            if i < tmp:
                epsilon += (1 - self._min_epsilon) * (tmp - i) / tmp
            return epsilon

        episode_rewards = []

        self._env._max_episode_steps = max_timesteps
        print(f'Env\'s maximum episode steps set to {self._env._max_episode_steps}.')

        nb_episodes += self._nb_training_steps
        print(f'Training starts (from {self._nb_training_steps} to {nb_episodes}).')

        for episode in tqdm.tqdm(range(self._nb_training_steps, nb_episodes)):
            # resets the environment
            obs = self._env.reset()
            done = False
            episode_reward = 0

            # epsilon is linearly decreasing given the training budget and exploration fraction if provided
            if exploration_ratio is not None:
                epsilon = compute_epsilon(episode, nb_episodes, exploration_ratio)
            # otherwise exploration-exploitation ratio is fixed to the minimum value
            else:
                epsilon = self._min_epsilon

            # monitors the training execution
            if (log_interval is not None) and ((episode + 1) % log_interval == 0):
                print(f'episode: {episode}, epsilon: {epsilon:.3f}, avg reward: {np.mean(episode_rewards[-10:]):0.1f}')
                self.save(f'taxi_large_map_qtable_{episode}.npy')

            while not done:
                # decides whether to explore or exploit
                if random.uniform(0, 1) > epsilon:
                    action = np.argmax(self._qtable[obs])
                else:
                    action = self._env.action_space.sample()

                # applies the action
                next_obs, reward, done, _ = self._env.step(action)

                # updates the table
                qvalue = self._qtable[obs, action]
                self._qtable[obs, action] = qvalue + self._lr * (reward + self._gamma * np.max(self._qtable[next_obs]) - qvalue)
                episode_reward += reward

                # updates the current observation
                obs = next_obs

            episode_rewards.append(episode_reward)

        score_frame = 10
        scores = [np.mean(episode_rewards[i:i+score_frame]) for i in range(min(len(episode_rewards), score_frame * (len(episode_rewards) // score_frame)))]
        result_filename = kwargs.get('result_filename', None)
        if result_filename is not None:
            fig, ax = plt.subplots()
            ax.plot(np.arange(1, len(scores) + 1), scores)
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Average episode rewards')
            fig.tight_layout()
            fig.savefig(f'{result_filename}_average_performance.png')
        print('Training done.')
        return episode_rewards


    def train_faulty(self, failure_target: int) -> Tuple[List[float], List[float]]:
        episode_rewards = []
        nb_episodes = 10000

        input_space = generate_input_space()
        eval_interval = 250
        evals = []
        def eval(inputs: np.ndarray):
            nb_faults = 0
            for input in inputs:
                obs = self._env.reset(input)
                failure = False
                for _ in range(200):
                    action = self.step(obs)
                    obs, reward, done, info = self._env.step(action)
                    if (reward == -10) or (info['crash'] == True):
                        failure = True
                        break
                    if done:
                        break
                nb_faults += int(failure)
            return nb_faults

        # training loop
        print('Training starts.')
        for episode in tqdm.tqdm(range(nb_episodes)):
            # evals the policy and stops the training according to @failure_target parameter
            if (episode + 1) % eval_interval == 0:
                nb_faults = eval(input_space)
                evals.append(nb_faults)
                if nb_faults <= failure_target:
                    break

            # resets the environment
            obs = self._env.reset()
            episode_reward = 0

            for _ in range(max_timesteps):
                # decides whether to explore or exploit
                if random.uniform(0, 1) > self._epsilon:
                    action = np.argmax(self._qtable[obs])
                else:
                    action = self._env.action_space.sample()

                # applies the action
                next_obs, reward, done, _ = self._env.step(action)
                # updates the table
                qvalue = self._qtable[obs, action]
                self._qtable[obs, action] = qvalue + self._lr * (reward + self._gamma * np.max(self._qtable[next_obs]) - qvalue)
                episode_reward += reward

                # updates the current observation
                obs = next_obs

                # ends episode if done
                if done == True:
                    break

            episode_rewards.append(episode_reward)
        print('Training done.')
        return episode_rewards, evals


    def step(self, obs: np.ndarray):
        return np.argmax(self._qtable[obs])


    def save(self, filename: str, erase: bool = False):
        '''Saves the values of the agent's Q-table as a numpy array.'''
        filename = f'{filename.split(".")[0]}.npy'

        if not erase and os.path.exists(filename):
            raise Exception('Files already exist. Save aborted.')

        # saves the q-table
        np.save(filename, self._qtable)


    def load(self, filename: str):
        '''Loads an agent.'''
        filename = f'{filename.split(".")[0]}.npy'

        if os.path.exists(filename):
            self._qtable = np.load(filename)
            match = re.search(r'(\d+)', filename)
            if match is not None:
                self._nb_training_steps = int(match.group(0))
            print(f'Qtable loaded; shape of {self._qtable.shape}.')
        else:
            raise FileNotFoundError(f'{filename} not found.')


########################################################################################################################


def test(policy: Agent, n: int = 100, visualize: bool = False):
    '''Evaluates the policy with random situations.'''
    env = gym.make('Taxi-v3')
    total_epochs, total_rewards, total_penalties = 0, 0, 0

    for _ in range(n):
        obs = env.reset()
        epochs, penalties, reward = 0, 0, 0
        if visualize:
            print(env.render(mode='ansi'))

        done = False
        total_reward = 0
        while not done:
            action = policy.step(obs)
            obs, reward, done, _info = env.step(action)
            total_reward += reward
            if visualize:
                print(env.render(mode='ansi'))
                input()
            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs
        total_rewards += total_reward

    print(f'Results after {n} episodes:')
    print(f'Average timesteps per episode: {total_epochs / n}')
    print(f'Average rewards per episode: {total_rewards / n}')
    print(f'Average penalties per episode: {total_penalties / n}')

    env.close()


def test_map(policy: Agent, map_fp: str, n: int = 10) -> Tuple[List[int], np.ndarray]:
    '''Tests the policy with random inputs.'''
    map = MapBuilder(map_fp)
    env = gym.make('Taxi-v3', map=map.map)
    input_space = map.generate_input_space()
    evals = []
    indices = np.random.choice(len(input_space), n)
    for i in tqdm.tqdm(indices):
        input = input_space[i]
        obs = env.reset(input)
        failure = False
        for _ in range(200):
            action = policy.step(obs)
            obs, reward, done, info = env.step(action)
            if (reward == -10) or (info['crash'] == True):
                failure = True
                break
            if done:
                break
        evals.append(int(failure))
    env.close()
    return evals, input_space[indices]


def eval_map(policy: Agent, map_fp: str) -> Tuple[List[int], np.ndarray]:
    '''Evaluates the policy over the entire input spacce.'''
    map = MapBuilder(map_fp)
    env = gym.make('Taxi-v3', map=map.map)
    input_space = map.generate_input_space()
    evals = []
    n = len(input_space)
    for i in tqdm.tqdm(range(n)):
        input = input_space[i]
        obs = env.reset(input)
        failure = False
        for _ in range(200):
            action = policy.step(obs)
            obs, reward, done, info = env.step(action)
            if (reward == -10) or (info['crash'] == True):
                failure = True
                break
            if done:
                break
        evals.append(int(failure))
    env.close()
    return evals, input_space


def execute_policy(policy: Agent, states: np.ndarray, visualize: bool = False, env: gym.Env = gym.make('Taxi-v3')):
    obs = env.reset(states)
    if visualize:
        print(env.render(mode='ansi'))
    timestep = 0
    penalty = 0
    episode_reward = 0
    done = False
    while not done:
        action = policy.step(obs)
        obs, reward, done, _info = env.step(action)
        episode_reward += reward
        if visualize:
            print(env.render(mode='ansi'))
            input()
        if reward == -10:
            penalty += 1
        timestep += 1
    print(timestep, penalty, episode_reward)


def reset_env_evaluation(map_fp: str, num_evals: int = 100_000):
    map = MapBuilder(map_fp)
    env = gym.make('Taxi-v3', map=map.map)
    input_space = [arr.tolist() for arr in map.generate_input_space()]
    print(f'Input space size: {len(input_space)}')
    resets = []
    for _ in tqdm.tqdm(range(num_evals)):
        resets.append(env.reset())
    env.close()
    u_resets = np.unique(resets)
    # checks whether the encodings are valid
    # encode_space = [env.encode(*input) for input in input_space]
    # assert np.all([state in encode_space for state in u_resets])
    print(f'{len(u_resets)} unique initial states among the {len(resets)} generated ones.')
    return u_resets


def evaluate_models(folder: str = 'models/') -> Tuple[List[str], List[List[int]]]:
    if not folder.endswith('/'):
        folder += '/'
    fps = [folder + f for f in os.listdir(folder)]
    evals_list = []
    for fp in fps:
        agent = TestAgent(fp)
        name = re.search(r'(\d+)', fp).group(0)
        evals = eval_map(agent, 'map_large.txt')[0]
        failure_rate = 100 * (sum(evals) / len(evals))
        print(f'agent {name} has a failure rate of {failure_rate:0.2f}%.')
        evals_list.append(evals)
    return fps, evals_list


# exec(open('taxi_agent.py').read())
if __name__ == '__main__':
    map = MapBuilder('map_large.txt')
    env = gym.make('Taxi-v3', map=map.map)

    nb_episodes = 1_500_000
    max_timesteps = 200
    learning_rate = 0.1
    gamma = 0.6
    epsilon = 0.1

    agent = ExplicitAgent(env, learning_rate, gamma, epsilon)
    # agent.load('taxi_large_map_qtable_2000000.npy')
    # evals, _ = eval_map(agent, 'map_large.txt')
    # print(f'failure rate: {(100 * sum(evals)/len(evals)):0.2}%')
    agent.train(nb_episodes, max_timesteps, log_interval=50_000, exploration_ratio=0.5)
    agent.save(f'taxi_large_map_qtable_{nb_episodes}.npy')