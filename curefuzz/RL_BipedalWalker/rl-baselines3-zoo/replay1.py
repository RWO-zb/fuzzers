import argparse
import importlib
import os
import sys
import pickle
import gym
import yaml
import numpy as np
import torch as th
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper
import copy # 我们需要 copy 来确保参数在循环中不被意外修改

# 导入与 enjoy_cure.py 相同的工具
import utils.import_envs
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.utils import StoreDict

def run_single_replay(base_args, crash_vector, index):
    """
    运行单次重放并保存数据的辅助函数。
    这个函数包含了模型加载和环境创建，以确保每次循环都是独立的。
    """
    
    # 打印当前进度
    print(f"\n" + "="*30)
    print(f"  Replaying Crash Vector {index}  ")
    print("="*30)
    
    # 复制参数，以防万一
    args = copy.deepcopy(base_args)

    # --- 1. 加载模型和环境 (与 replay1.py 逻辑相同) ---
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    model_path = os.path.join(log_path, f"{env_id}.zip")
    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")

    assert os.path.isfile(model_path), f"No model found for {algo} on {env_id}, path: {model_path}"

    set_random_seed(args.seed)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=None,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    
    # 修复 Assertion Error 的代码
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    
    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
    print(f"Model and Env for Vector {index} loaded.")

    # --- 2. 运行重放并存储数据 (使用传入的 vector 和 index) ---
    
    observation_sequence = []
    action_sequence = []
    reward_sequence = []
    
    # 使用崩溃向量重置环境
    obs = env.reset(crash_vector)
    state = None
    observation_sequence.append(obs[0]) 

    for step in range(args.n_timesteps):
        action, state = model.predict(obs, state=state, deterministic=args.deterministic)
        obs, reward, done, infos = env.step(action)
        
        observation_sequence.append(obs[0])
        action_sequence.append(action[0])
        reward_sequence.append(reward[0])

        if not args.no_render:
            env.render("human")

        if done:
            print(f"Crash replayed. Episode finished at step {step + 1}.")
            break

    env.close()

    # --- 3. 将数据打包并保存到文件 ---
    replay_data = {
        'crash_vector': crash_vector,
        'observation_sequence': np.array(observation_sequence),
        'action_sequence': np.array(action_sequence),
        'reward_sequence': np.array(reward_sequence),
        'crash_step': len(reward_sequence),
        'total_reward': np.sum(reward_sequence)
    }
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
        
    # 构建动态的文件名
    output_filename = f"crash_{index}_data.pkl"
    output_path = os.path.join(args.output_dir, output_filename)
        
    with open(output_path, 'wb') as f:
        pickle.dump(replay_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Successfully saved replay data for vector {index} to: {output_path}")


def main():
    # --- 参数解析 ---
    # 我们保留所有模型/环境参数，但修改文件参数
    
    parser = argparse.ArgumentParser()
    
    # --- 复制 enjoy_cure.py 的参数 ---
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="../rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=300, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    # --- 结束复制 ---

    # --- 修改为批量处理的参数 ---
    parser.add_argument(
        "--crash-file",
        help="Path to the .pkl file containing ALL crash vectors (e.g., cure_crash.pkl)",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output-dir",
        help="Path to the DIRECTORY where all replay .pkl files will be saved",
        type=str,
        default="replay_results" # 默认保存到 'replay_results/' 文件夹
    )
    
    args = parser.parse_args()

    # --- 批量处理的主循环 ---
    print(f"Loading all crash vectors from: {args.crash_file}")
    try:
        with open(args.crash_file, 'rb') as f:
            all_crashes = pickle.load(f)
        print(f"Successfully loaded {len(all_crashes)} crash vectors.")
    except FileNotFoundError:
        print(f"Error: Input crash file not found at {args.crash_file}")
        return
    except Exception as e:
        print(f"Error loading {args.crash_file}: {e}")
        return

    # 遍历所有加载的崩溃向量
    for index, crash_vector in enumerate(all_crashes):
        # 调用辅助函数来处理每一个向量
        # 这样做可以确保每轮循环的环境和模型都是“干净”的
        run_single_replay(args, crash_vector, index)
    
    print("\n" + "="*30)
    print("  All replay tasks finished.  ")
    print(f"Data saved in directory: {args.output_dir}")
    print("="*30)


if __name__ == "__main__":
    main()