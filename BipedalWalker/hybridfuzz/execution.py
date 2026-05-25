import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ExecutionResult:
    input: np.ndarray
    reward: float
    last_reward: float
    done: bool
    did_physical_crash: bool
    is_reward_fault: bool
    is_fault: bool
    obs_seq: List[np.ndarray]
    raw_transitions: List[Any]
    behavior_features: List[float]
    qd_behavior: List[float]
    final_state: List[float]
    survival_steps: int
    execution_cost: float
    env_name: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "reward": self.reward,
            "last_reward": self.last_reward,
            "done": self.done,
            "did_physical_crash": self.did_physical_crash,
            "is_reward_fault": self.is_reward_fault,
            "is_fault": self.is_fault,
            "is_crash": self.is_fault,
            "physical_crash": self.did_physical_crash,
            "reward_fault": self.is_reward_fault,
            "obs_seq": self.obs_seq,
            "raw_transitions": self.raw_transitions,
            "behavior_features": self.behavior_features,
            "qd_behavior": self.qd_behavior,
            "final_state": self.final_state,
            "survival_steps": self.survival_steps,
            "execution_cost": self.execution_cost,
            "env_name": self.env_name,
        }


def unwrap_env(env):
    current = env
    while hasattr(current, "venv"):
        current = current.venv
    if hasattr(current, "envs") and len(current.envs) > 0:
        current = current.envs[0]
    while hasattr(current, "env"):
        current = current.env
    return getattr(current, "unwrapped", current)


def scalar(value, default=0.0):
    if isinstance(value, (list, tuple)):
        return scalar(value[0], default)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return scalar(value.reshape(-1)[0], default)
    try:
        return float(value)
    except Exception:
        return default


def bool_scalar(value) -> bool:
    if isinstance(value, (list, tuple)):
        return bool_scalar(value[0]) if value else False
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0]) if value.size else False
    return bool(value)


def first_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    arr = np.asarray(obs)
    if arr.ndim > 1:
        return arr[0].copy()
    return arr.copy()


class UnifiedExecutor:
    def __init__(
        self,
        default_env,
        model,
        predict_fn,
        n_timesteps: int,
        reward_fault_threshold: float = 10.0,
        qdfuzz_env: Optional[Any] = None,
        qd_descriptors: Optional[List[int]] = None,
    ):
        self.default_env = default_env
        self.qdfuzz_env = qdfuzz_env
        self.model = model
        self.predict_fn = predict_fn
        self.n_timesteps = n_timesteps
        self.reward_fault_threshold = reward_fault_threshold
        self.qd_descriptors = qd_descriptors or [4, 8]

    def env_for_strategy(self, strategy_name: str):
        if strategy_name == "qdfuzz" and self.qdfuzz_env is not None:
            return self.qdfuzz_env, "qdfuzz-v4"
        return self.default_env, "default-v3"

    def execute(self, candidate, strategy_name: str) -> ExecutionResult:
        env, env_name = self.env_for_strategy(strategy_name)
        candidate = np.asarray(candidate, dtype=int)

        start_time = time.time()
        obs = env.reset(candidate)
        current_obs = first_obs(obs)
        obs_seq = [current_obs]
        raw_transitions = []
        feature_sum = None
        total_reward = 0.0
        last_reward = 0.0
        done_flag = False
        steps = 0

        for _ in range(self.n_timesteps):
            if self.model is not None:
                action = self.predict_fn(self.model, obs)
            else:
                sampled = env.action_space.sample()
                action = sampled if env_name == "qdfuzz-v4" else [sampled]

            next_obs, reward, done, info = env.step(action)
            reward_value = scalar(reward)
            done_flag = bool_scalar(done)
            next_obs_0 = first_obs(next_obs)

            raw_transitions.append((current_obs.copy(), np.asarray(action).copy(), reward_value, next_obs_0.copy(), done_flag))

            info_obj = info[0] if isinstance(info, (list, tuple)) and info else info
            if isinstance(info_obj, dict) and "features" in info_obj:
                features = np.asarray(info_obj["features"], dtype=float)
                feature_sum = features.copy() if feature_sum is None else feature_sum + features

            total_reward += reward_value
            last_reward = reward_value
            steps += 1
            obs_seq.append(next_obs_0)
            obs = next_obs
            current_obs = next_obs_0

            if done_flag:
                break

        execution_cost = time.time() - start_time
        base_env = unwrap_env(env)
        game_over = bool(getattr(base_env, "game_over", False))
        did_physical_crash = bool(done_flag or game_over or last_reward == -100)
        is_reward_fault = bool((not did_physical_crash) and (total_reward < self.reward_fault_threshold))
        is_fault = did_physical_crash or is_reward_fault

        if feature_sum is None:
            behavior_features = np.zeros(12, dtype=float)
        else:
            behavior_features = feature_sum / max(1, steps)

        qd_behavior = behavior_features[self.qd_descriptors].tolist()
        final_state = obs_seq[-1].tolist() if obs_seq else []

        return ExecutionResult(
            input=candidate,
            reward=float(total_reward),
            last_reward=float(last_reward),
            done=done_flag,
            did_physical_crash=did_physical_crash,
            is_reward_fault=is_reward_fault,
            is_fault=is_fault,
            obs_seq=obs_seq,
            raw_transitions=raw_transitions,
            behavior_features=behavior_features.tolist(),
            qd_behavior=qd_behavior,
            final_state=final_state,
            survival_steps=steps,
            execution_cost=execution_cost,
            env_name=env_name,
        )
