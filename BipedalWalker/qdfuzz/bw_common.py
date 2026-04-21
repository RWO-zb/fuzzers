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
main_seed = 723
MIN_INPUT = np.array([1 for _ in range(15)])
MAX_INPUT = np.array([3 for _ in range(15)])
MAX_DIST_INPUT: np.ndarray = np.linalg.norm(MAX_INPUT - MIN_INPUT)
AVG_SIZE = 30
EXPERT_INDICES = [
    [4, 8]
]
EXPERT_PLOT_ARGS = [
    {'xlabel': 'distance to the goal', 'ylabel': 'hull angle', 'title': 'Distance vs Hull angle'},
    {'xlabel': 'torque (actions)', 'ylabel': 'jump rate', 'title': 'Torque vs Jump'},
    {'xlabel': '1st leg', 'ylabel': '2nd leg', 'title': 'Hip angles'},
    {'xlabel': '1st leg', 'ylabel': '2nd leg', 'title': 'Hip speeds'}
]

###################### EXECUTION/EXPERIMENT SUPPORTERS ################################

# [新增] 极速推理优化: 直接提取网络均值输出，彻底规避 distribution.py 产生的计算开销。
def fast_predict(model, obs):
    # [修复] 原生 Gym 返回 1D obs，使用 .reshape(1, -1) 补充网络所需的 batch 维度
    obs_tensor = torch.as_tensor(obs).reshape(1, -1).float().to("cpu")
    
    with torch.no_grad():
        # 针对 BipedalWalker 常用的 TQC / SAC / TD3 算法
        if hasattr(model.policy, "actor") and hasattr(model.policy.actor, "get_action_dist_params"):
            mean_actions, _, _ = model.policy.actor.get_action_dist_params(obs_tensor)
            # 动作必须经过 tanh 映射到 [-1, 1]
            action = torch.tanh(mean_actions).cpu().numpy()
            
            # [修复] 去除 batch 维度，返回原版 step 期望的 1D 数组 action
            return action[0]
            
        # 兜底：如果算法不匹配，使用原版 predict
        return model.predict(obs, deterministic=True)[0]


def generate_input(rng: np.random.Generator = None):
    if rng is None:
        return np.random.randint(low=1, high=4, size=15)
    else:
        return rng.integers(low=1, high=4, size=15)


def generate_inputs(rng: np.random.Generator, n: int):
    return rng.integers(low=1, high=4, size=n)


def load_model():
    return TQC.load('../rl-trained-agents/tqc/BipedalWalkerHardcore-v3_1/BipedalWalkerHardcore-v3.zip',
                    custom_objects={
                        "learning_rate": lambda _: 3e-4,
                        "lr_schedule": lambda _: 3e-4,
                     }, 
                    kwargs={'seed': 0, 'buffer_size': 1},
                    device="cpu") # [新增] 极速推理优化: 强制指定加载到 CPU 进行推断


def get_key(input: np.ndarray):
    '''Integer like representation of the float numpy arrays as keys.'''
    return ' '.join([f'{i:.0f}' for i in input])


def get_input_from_key(key: str) -> np.ndarray:
    return np.asfarray(key.split(' '), dtype=str).astype(int)


def get_inputs_from_keys(keys: Iterable[str]) -> np.ndarray:
    return np.array([np.asfarray(k.split(' '), dtype=str).astype(int) for k in keys])


# [修改] 返回值类型注解增加了 eval_info (dict) 以对齐评估指标
def execute_policy(input: np.ndarray, model: BaseAlgorithm, env_seed: int, descriptors: List = None, sim_steps: int = 300) -> Tuple:
    '''Executes the model on the environment and only computes the 12 features used by Leo Cazenille. It also returns the final state.'''

    env = gym.make('BipedalWalkerHardcore-v4',rand_seed=main_seed)

    acc_reward = 0.0
    features = np.zeros(12)

    # [新增] 对齐评估指标: 记录环境初始化时间
    env_sim_time = 0.0
    t_sim_start = time.time()
    obs = env.reset(input)
    env_sim_time += time.time() - t_sim_start

    t0 = time.time()
    
    # [新增] 存储 TodyNet 所需的 (State+Action) 序列
    todynet_trace = []
    
    # [新增] 存储 RL 重训所需的 (s, a, r, s', d) 序列
    rl_transitions = []

    # [新增] 对齐评估指标: 提取仅所需的轨迹数据 (obs[0])
    obs_0_trajectory = [obs.copy()] 

    for t in range(sim_steps):
        # [修改] 极速推理优化: 替换原有的 model.predict 为 fast_predict
        action = fast_predict(model, obs)
        
        # [新增] 收集 TodyNet 数据 (在 step 之前收集当前的 obs 和即将执行的 action)
        todynet_trace.append(np.concatenate([obs.flatten(), action.flatten()]))
        
        # [新增] 对齐评估指标: 记录环境步进时间
        t_sim_start = time.time()
        next_obs, reward, done, info = env.step(action)
        env_sim_time += time.time() - t_sim_start
        
        # [新增] 收集 RL Transition 数据 (Raw Obs, Action, Reward, Raw Next Obs, Done)
        # 注意：这里的 env 是 raw gym env，所以 obs 已经是 raw 的，无需反归一化
        rl_transitions.append((obs, action, reward, next_obs, done))
        
        # [新增] 对齐评估指标: 记录观测轨迹
        obs_0_trajectory.append(next_obs.copy())

        features += info['features'] # numpy array
        acc_reward += reward
        
        obs = next_obs # 更新 obs

        if done:
            break

    env.close()
    if t > 0:
        features /= t
    exec_time = time.time() - t0

    # 修改判定逻辑：如果最后一步reward是-100（摔倒）或者总奖励 acc_reward < 10，则判定为Crash
    is_crash = (reward == -100) or (acc_reward < 10)

    # [新增] 对齐评估指标: 将 done 和 threshold 剥离并严格互斥
    did_crash = bool(done)
    is_reward_fault = False
    if not did_crash and is_crash:
        is_reward_fault = True

    # [新增] 对齐评估指标: 构建统一日志字典
    eval_info = {
        'did_crash': did_crash,
        'is_reward_fault': is_reward_fault,
        'survival_steps': t + 1,
        'output_trajectory': np.array(obs_0_trajectory, dtype=np.float32) if (did_crash or is_reward_fault) else None,
        'env_sim_time': env_sim_time
    }

    # [修改] 返回值增加了 eval_info
    if descriptors is not None:
        descriptors = np.array(descriptors)
        assert all(descriptors < 12) and all(descriptors >= 0)
        return acc_reward, is_crash, features[descriptors], obs, exec_time, todynet_trace, rl_transitions, eval_info
    else:
        return acc_reward, is_crash, features, obs, exec_time, todynet_trace, rl_transitions, eval_info


def execute_policy_trajectory(input: np.ndarray, model: BaseAlgorithm, env_seed: int, sim_steps: int = 300) -> Tuple[float, bool, np.ndarray, List[np.ndarray], float]:
    '''Executes the model and returns the trajectory data. Useful for MDPFuzz.'''
    env = gym.make('BipedalWalkerHardcore-v3', rand_seed=env_seed)
    features = np.zeros(12)
    obs_seq = []
    acc_reward = 0.0
    done=False
    obs = env.reset(input)
    
    t0 = time.time()
    for t in range(sim_steps):
        # [修改] 极速推理优化: 替换原有的 model.predict 为 fast_predict
        action = fast_predict(model, obs)
        
        obs, reward, done, info = env.step(action)
        features += info['features'] # numpy array
        acc_reward += reward
        obs_seq.append(obs)
        if done:
            break

    env.close()
    if t > 0:
        features /= t
    exec_time = time.time() - t0
    
    # 修改判定逻辑：同上
    is_crash = (reward == -100) or (acc_reward < 10)

    return acc_reward, is_crash, features, np.array(obs_seq), exec_time


def get_edges(env_seed: int, descriptors: np.ndarray, sim_steps: int = 300) -> np.ndarray:
    '''Returns the saved grid edges.'''
    edges = np.load(f'grid/bw/{env_seed}_{sim_steps}_edges.npy')
    return edges[descriptors]


if __name__ == '__main__':
    # [新增] 极速推理优化: 将内部线程限制为1以最大化单核性能
    torch.set_num_threads(1)
    
    main_seed = 2021
    env_seed = 0
    model = load_model()

    rng: np.random.Generator = np.random.default_rng(main_seed)
    descriptors = EXPERT_INDICES[0]
    oracles, rewards, behaviors, final_states = [], [], [], []

    for _ in tqdm.tqdm(range(100)):
        input: np.ndarray = rng.integers(low=1, high=4, size=15)
        # [修改] 接收 8 个返回值 (包含 eval_info)
        r, o, b, fs, _, _, _, _ = execute_policy(input, model, env_seed, descriptors, 1000)
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