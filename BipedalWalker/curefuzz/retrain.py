import argparse
import os
import pickle
import sys
import numpy as np
import yaml
import importlib
import random
import torch

from stable_baselines3.common.utils import set_random_seed, get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

# 引入项目现有的工具
from utils import ALGOS, create_test_env, get_saved_hyperparams
from utils.utils import StoreDict

def main():
    parser = argparse.ArgumentParser(description="Retrain Raw Model (Matches config.yml)")
    parser.add_argument("--env", help="environment ID", type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument("--algo", help="RL Algorithm", default="tqc", type=str, choices=list(ALGOS.keys()))
    parser.add_argument("--model-path", help="Path to the .zip model file", type=str, required=True)
    parser.add_argument("--transitions-path", help="Path to transitions.pkl", type=str, required=True)
    
    # 训练参数
    parser.add_argument("-n", "--n-timesteps", help="Online training steps", default=0, type=int)
    parser.add_argument("--offline-steps", help="Offline gradient steps", default=50000, type=int)
    parser.add_argument("--lr", help="Learning rate", default=1e-7, type=float)
    parser.add_argument("--crash-weight", help="Weight for crash samples", default=10, type=int)
    
    parser.add_argument("--seed", help="Random seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--gym-packages", type=str, nargs="+", default=[], help="External Gym packages")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Env kwargs")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs/retrain_raw")
    
    args = parser.parse_args()

    # 1. 算法检查
    if args.algo not in ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]:
        print(f"Error: Algorithm {args.algo} is not Off-Policy.")
        sys.exit(1)

    for env_module in args.gym_packages:
        importlib.import_module(env_module)
    set_random_seed(args.seed)

    # 2. 准备环境 (严格匹配 config.yml)
    model_dir = os.path.dirname(args.model_path)
    stats_path = os.path.join(model_dir, args.env)
    
    # 获取原有超参数
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=False)
    
    # [CRITICAL] 你的 config.yml 没有 normalize，这里确保它被设为 False
    if hyperparams.get('normalize') is True:
        print("[Warning] config.yml says Normalize=True but file missing? Forcing False based on user input.")
    
    hyperparams['normalize'] = False
    print("[Info] Configuration: Normalization is DISABLED (Raw Mode).")

    env_kwargs = {}
    args_path = os.path.join(model_dir, args.env, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f_args:
            loaded_args = yaml.load(f_args, Loader=yaml.UnsafeLoader)
            if loaded_args.get("env_kwargs") is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    print(f"Creating environment {args.env}...")
    env = create_test_env(
        args.env,
        n_envs=1, 
        stats_path=None, # 故意设为 None，防止它去读不存在的 pkl
        seed=args.seed,
        log_dir=None,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    
    # 双重检查：剥离任何意外添加的 VecNormalize
    if isinstance(env, VecNormalize):
        print("[Warning] Removing unexpected VecNormalize wrapper...")
        env = env.venv

    # 3. 加载模型
    print(f"Loading model from {args.model_path}...")
    # 使用线性衰减的学习率
    lr_schedule = get_linear_fn(args.lr, args.lr * 0.1, 1.0)
    custom_objects = {
        "learning_rate": lr_schedule,
        "lr_schedule": lr_schedule,
        "clip_range": lambda _: 0.0, 
    }

    model = ALGOS[args.algo].load(args.model_path, env=env, custom_objects=custom_objects)
    
    # 允许所有层更新 (全量微调，因为是 Raw 模式)
    for param in model.actor.parameters():
        param.requires_grad = True
    for param in model.critic.parameters():
        param.requires_grad = True

    if hasattr(model, 'learning_starts'):
        model.learning_starts = 0

    # 4. 数据注入
    print(f"Loading transitions from {args.transitions_path}...")
    with open(args.transitions_path, 'rb') as f:
        data = pickle.load(f)
    
    weighted_stream = []
    if isinstance(data, dict):
        crash_list = data.get("crash", [])
        success_list = data.get("success", [])
        print(f"  - Crash samples: {len(crash_list)} (Weight={args.crash_weight})")
        print(f"  - Success samples: {len(success_list)}")
        for t in crash_list:
            weighted_stream.append((t, args.crash_weight))
        for t in success_list:
            weighted_stream.append((t, 1))
    else:
        for t in data:
            weighted_stream.append((t, 1))

    random.shuffle(weighted_stream)

    print("Injecting RAW transitions into Replay Buffer...")
    for transition, weight in weighted_stream:
        # 数据直接注入，不需 normalize_obs
        obs, action, reward, next_obs, done = transition
        
        # 维度调整
        obs_ = obs.reshape(1, *obs.shape) if isinstance(obs, np.ndarray) else np.array([obs])
        next_obs_ = next_obs.reshape(1, *next_obs.shape) if isinstance(next_obs, np.ndarray) else np.array([next_obs])
        action_ = action.reshape(1, *action.shape) if isinstance(action, np.ndarray) else np.array([action])
        reward_ = np.array([reward])
        done_ = np.array([done])
        infos_ = [{}]
        
        for _ in range(weight):
            model.replay_buffer.add(obs_, next_obs_, action_, reward_, done_, infos_)

    print(f"Buffer Size: {model.replay_buffer.size()}")

    # 5. 设置日志
    new_log_path = os.path.join(args.folder, f"{args.algo}_raw_{args.seed}")
    os.makedirs(new_log_path, exist_ok=True)
    new_logger = configure(new_log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # 6. 训练流程
    if model.replay_buffer.size() < model.batch_size:
        print("Filling buffer...")
        model.learn(total_timesteps=(model.batch_size - model.replay_buffer.size()))

    print(f"Starting Offline Pre-training ({args.offline_steps} steps)...")
    if args.offline_steps > 0:
        model.train(gradient_steps=args.offline_steps, batch_size=model.batch_size)

    print(f"Starting Online Retraining ({args.n_timesteps} steps)...")
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=new_log_path, name_prefix="raw_finetune")
    model.learn(total_timesteps=args.n_timesteps, callback=checkpoint_callback)

    # 7. 保存结果
    save_name = f"{args.env}_retrained_final"
    save_path = os.path.join(new_log_path, save_name)
    model.save(save_path)
    
    print("-" * 50)
    print(f"Model saved to: {save_path}.zip")
    print("NOTE: This model is RAW mode (matches your original config).")
    print("It does NOT require vecnormalize.pkl.")
    print("-" * 50)

if __name__ == "__main__":
    main()