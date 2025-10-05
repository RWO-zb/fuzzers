import gymnasium as gym
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from copy import deepcopy
import numpy as np

class StoreAndTerminateWrapper(gym.Wrapper):
    '''
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    '''
    def __init__(self, env):
        super(StoreAndTerminateWrapper,self).__init__(env)
        self.max_steps = 200
        self.current_step = 0
        self.env=env
        self.mem = []
        self.TotalReward = 0.0
        self.first_state = 0
        self.first_obs = 0
        self.prev_obs = 0
        self.states_list = []
        self.info = {}

    def reset(self, *args, **kwargs):
        self.current_step = 0
        obs, info = self.env.reset(*args, **kwargs)
        self.TotalReward = 0.0
        self.first_obs = obs
        return obs,info

    def step(self, action):
        if self.current_step == 0:
            self.prev_obs = self.first_obs
            self.first_state = deepcopy(self.env)
            self.states_list.append(self.first_state)
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.TotalReward += reward
        self.mem.append(tuple((self.prev_obs,action)))
        self.prev_obs = obs
        if self.current_step >= self.max_steps:
            truncated = True
        if obs[0] <= -1.2:
            truncated = True
            reward = -201 - self.TotalReward
            self.TotalReward = -200
        if terminated or truncated:
            self.mem.append(tuple(('done',self.TotalReward)))
        self.info['mem'] = self.mem
        self.info['state'] = self.states_list
        return obs, reward, terminated, truncated, info

    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        self.current_step = 0
        self.TotalReward = 0.0
        self.first_obs = obs
        return obs

class StopOnFailureRateCallback(BaseCallback):
    """
    Callback for stopping training once the agent has reached a specific failure rate threshold.

    Args:
        eval_env (gym.Env): The environment used for evaluation.
        eval_frequency (int): Frequency of the callback.
        num_eval_episodes (int, optional): The number of episodes to approximate the failure rate of the agent. Default to 100.
        failure_rate_treshold (float, optional): Failure rate threshold to stop training. Default to 0.10.
        start_on_steps (int, optional): Steps after which the callback will be called. Defaults to 0.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_frequency: int,
        num_eval_episodes=int,
        failure_rate_treshold=0.10,
        start_on_steps=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eval_env = eval_env
        self.eval_frequency = eval_frequency
        self.num_eval_episodes = num_eval_episodes
        self.failure_rate_treshold = failure_rate_treshold
        self.start_on_steps = start_on_steps

    def _on_step(self):

        if self.n_calls >= self.start_on_steps and (
            self.n_calls % self.eval_frequency == 0
        ):
            failure_rate = self._evaluate_failure_rate()
            if self.verbose > 0:
                print(
                    "Failure rate at steps {}: {:.2%}.".format(
                        self.n_calls, failure_rate
                    )
                )
            if failure_rate <= self.failure_rate_treshold:
                return False
        return True

    def _evaluate_failure_rate(self) -> float:
        num_failures = 0
        for _ in range(self.num_eval_episodes):
            obs, info = self.eval_env.reset()
            terminated = truncated = False
            state = False
            while not (terminated or truncated):
                action, state = self.model.predict(obs, state=state, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
            num_failures += int(terminated == False)
        failure_rate = num_failures / self.num_eval_episodes
        return failure_rate
    
mtc = gym.make('MountainCar-v0')
env = StoreAndTerminateWrapper(mtc)
dqn_model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    train_freq=16,
    gradient_steps=8,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.07,
    target_update_interval=600,
    learning_starts=1000,
    buffer_size=10000,
    batch_size=128,
    learning_rate=4e-3,
    policy_kwargs=dict(net_arch=[256, 256]),
)

callback = StopOnFailureRateCallback(
    eval_env=gym.make("MountainCar-v0", render_mode="rgb_array"),
    eval_frequency=500,
    num_eval_episodes=500,
    failure_rate_treshold=0.20,
    verbose=1,
    start_on_steps=90_000,
)

dqn_model.learn(total_timesteps=120_000, callback=callback)
print("Training stops after {} steps.".format(callback.n_calls))
MODEL_PATH = "1.zip"
dqn_model.save(MODEL_PATH)