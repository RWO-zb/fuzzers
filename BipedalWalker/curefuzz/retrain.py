import argparse
import os
import pickle
import sys
import numpy as np
import yaml
import importlib
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# 新增：引入 logger 配置函数
from stable_baselines3.common.logger import configure

# 引入项目现有的工具
from utils import ALGOS, create_test_env, get_saved_hyperparams
from utils.utils import StoreDict

def main():
    parser = argparse.ArgumentParser(description="Retrain agent using collected transitions")
    parser.add_argument("--env", help="environment ID", type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument("--algo", help="RL Algorithm", default="tqc", type=str, choices=list(ALGOS.keys()))
    parser.add_argument("--model-path", help="Path to the .zip model file to retrain", type=str, required=True)
    parser.add_argument("--transitions-path", help="Path to the transitions.pkl file", type=str, required=True)
    parser.add_argument("-n", "--n-timesteps", help="Number of timesteps to retrain", default=100000, type=int)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true", default=False, help="Do not render")
    parser.add_argument("--gym-packages", type=str, nargs="+", default=[], help="External Gym packages")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Env constructor kwargs")
    parser.add_argument("-f", "--folder", help="Log folder (for saving new model)", type=str, default="logs/retrain")
    
    args = parser.parse_args()

    # 1. 检查算法类型
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
    if args.algo not in off_policy_algos:
        print(f"Error: Algorithm {args.algo} is not Off-Policy. Pre-filling replay buffer only works for Off-Policy algorithms.")
        sys.exit(1)

    # 2. 导入自定义 Gym 环境
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    # 3. 设置随机种子
    set_random_seed(args.seed)

    # 4. 准备环境参数 (尝试从模型所在目录加载原有配置)
    model_dir = os.path.dirname(args.model_path)
    stats_path = os.path.join(model_dir, args.env)
    
    # 获取之前保存的超参数和 VecNormalize 统计数据
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=False)

    env_kwargs = {}
    args_path = os.path.join(model_dir, args.env, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f_args:
            loaded_args = yaml.load(f_args, Loader=yaml.UnsafeLoader)
            if loaded_args.get("env_kwargs") is not None:
                env_kwargs = loaded_args["env_kwargs"]
    
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    # 5. 创建环境
    print(f"Creating environment {args.env}...")
    env = create_test_env(
        args.env,
        n_envs=1, # Off-policy 通常使用 n_envs=1
        stats_path=stats_path, # 加载之前的归一化参数
        seed=args.seed,
        log_dir=None,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    
    # 确保环境处于训练模式 (更新 moving average)
    if isinstance(env, VecNormalize):
        env.training = True
        env.norm_reward = True 

    # 6. 加载模型
    print(f"Loading model from {args.model_path}...")
    custom_objects = {}
    # 处理 Python 版本兼容性问题（如果需要）
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    
    model = ALGOS[args.algo].load(args.model_path, env=env, custom_objects=custom_objects)

    # 7. 加载并注入 Transitions 数据
    print(f"Loading transitions from {args.transitions_path}...")
    with open(args.transitions_path, 'rb') as f:
        transitions = pickle.load(f)
    
    print(f"Found {len(transitions)} transitions. Injecting into Replay Buffer...")
    
    # 检查 Replay Buffer 是否存在
    if model.replay_buffer is None:
        print("Error: Model does not have a replay buffer initialized.")
        sys.exit(1)

    initial_buffer_size = model.replay_buffer.size()
    print(f"Initial buffer size: {initial_buffer_size}")

    count = 0
    for obs, action, reward, next_obs, done in transitions:
        # SB3 的 replay buffer add 方法通常期望输入维度为 (n_envs, dim)
        # 即使 n_envs=1，我们也需要 reshape 成 (1, ...)
        
        # 处理 Observation
        if isinstance(obs, np.ndarray):
            obs_ = obs.reshape(1, *obs.shape)
        else:
            obs_ = np.array([obs])
            
        # 处理 Next Observation
        if isinstance(next_obs, np.ndarray):
            next_obs_ = next_obs.reshape(1, *next_obs.shape)
        else:
            next_obs_ = np.array([next_obs])
            
        # 处理 Action
        if isinstance(action, np.ndarray):
            action_ = action.reshape(1, *action.shape)
        else:
            action_ = np.array([action])
            
        # 处理 Reward
        reward_ = np.array([reward])
        
        # 处理 Done
        done_ = np.array([done])
        
        # Infos (通常为空，但 add 需要此参数)
        infos_ = [{}]
        
        try:
            model.replay_buffer.add(obs_, next_obs_, action_, reward_, done_, infos_)
            count += 1
        except Exception as e:
            print(f"Error adding transition {count}: {e}")
            break

    print(f"Successfully added {count} transitions.")
    print(f"New buffer size: {model.replay_buffer.size()}")

    # 8. 开始再训练 (Retrain)
    print(f"Starting retraining for {args.n_timesteps} steps...")
    
    # 设置新的日志路径
    new_log_path = os.path.join(args.folder, f"{args.algo}_retrained_{args.seed}")
    os.makedirs(new_log_path, exist_ok=True)
    
    # --- [关键修改点] ---
    # 显式配置 Logger，而不是设为 None
    print(f"Configuring new logger to {new_log_path}...")
    new_logger = configure(new_log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    # -------------------
    
    try:
        model.learn(total_timesteps=args.n_timesteps)
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    # 9. 保存新模型
    save_name = f"{args.env}_retrained_{args.n_timesteps}_steps"
    save_path = os.path.join(new_log_path, save_name)
    print(f"Saving retrained model to {save_path}.zip")
    model.save(save_path)
    
    # 如果使用了 VecNormalize，也需要保存其统计数据
    if isinstance(env, VecNormalize):
        norm_path = os.path.join(new_log_path, f"{args.env}", "vecnormalize.pkl")
        os.makedirs(os.path.dirname(norm_path), exist_ok=True)
        env.save(norm_path)
        print(f"Saved VecNormalize stats to {norm_path}")

    print("Done.")

if __name__ == "__main__":
    main()