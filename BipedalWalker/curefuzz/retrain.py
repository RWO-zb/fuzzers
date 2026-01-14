import argparse
import os
import pickle
import sys
import numpy as np
import yaml
import importlib
import gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnvWrapper
from stable_baselines3.common.logger import configure

# 引入项目现有的工具
from utils import ALGOS, create_test_env, get_saved_hyperparams
from utils.utils import StoreDict

# ==========================================
# [新增] 1. 物理传送函数 (直接内嵌)
# ==========================================
def teleport_robot(env, state_dict):
    """
    将 BipedalWalker 机器人的物理状态强制设置为 state_dict 中的值
    """
    # 尝试获取底层环境 (Box2D)
    # VecEnv -> Monitor -> TimeLimit -> BipedalWalker
    base_env = env.unwrapped
    
    if not hasattr(base_env, 'hull') or not hasattr(base_env, 'legs'):
        return False

    # 1. 恢复躯干 (Hull)
    h = base_env.hull
    h.position = state_dict["hull_pos"]
    h.angle = state_dict["hull_angle"]
    h.linearVelocity = state_dict["hull_lin_vel"]
    h.angularVelocity = state_dict["hull_ang_vel"]
    h.awake = True # 唤醒物理引擎
    
    # 2. 恢复腿部 (Legs)
    # state_dict["legs"] 顺序对应 base_env.legs 的列表顺序
    for i, leg in enumerate(base_env.legs):
        if i < len(state_dict["legs"]):
            s = state_dict["legs"][i]
            leg.position = s["pos"]
            leg.angle = s["angle"]
            leg.linearVelocity = s["lin_vel"]
            leg.angularVelocity = s["ang_vel"]
            leg.awake = True
            
    return True

# ==========================================
# [新增] 2. 课程学习 Wrapper (适配 VecEnv)
# ==========================================
class PhysicsCurriculumVecWrapper(VecEnvWrapper):
    """
    针对 VecEnv 的反向课程 Wrapper。
    在 reset() 时，以 sample_prob 的概率将子环境传送到 Crash 前夕的状态。
    """
    def __init__(self, venv, physics_file, sample_prob=0.2, lookback_window=(20, 60)):
        super().__init__(venv)
        self.physics_file = physics_file
        self.sample_prob = sample_prob
        self.lookback_window = lookback_window
        self.crash_data = []
        
        # 加载数据
        if os.path.exists(self.physics_file):
            print(f"[Curriculum] Loading physics data from {self.physics_file}...")
            with open(self.physics_file, 'rb') as f:
                self.crash_data = pickle.load(f)
            print(f"[Curriculum] Loaded {len(self.crash_data)} trajectories.")
        else:
            print(f"[Curriculum] Warning: File {self.physics_file} not found. Curriculum disabled.")

    def reset(self):
        # 1. 先执行正常的 Reset (生成地形、重置内部计数器等)
        obs = self.venv.reset()
        
        # 如果没有数据，直接返回
        if not self.crash_data:
            return obs

        # 2. 遍历所有子环境 (通常 off-policy 只有 1 个 env)
        # 我们需要修改 obs 中的数据
        for i in range(self.num_envs):
            # 以一定概率触发“困难模式”
            if np.random.rand() < self.sample_prob:
                
                # A. 随机选择一条 Crash 轨迹
                case = self.crash_data[np.random.randint(len(self.crash_data))]
                # seed = case['seed'] # 地形 Seed 已经在 venv.reset() 中处理了吗？
                # 注意：VecEnv 的 reset 通常不接受 seed 参数。
                # 如果我们要严格复现地形，需要手动设置 seed。
                # 这里的 venv.reset() 已经随机生成了地形，如果地形和 Crash 时差别太大，传送可能会导致穿模。
                # 但 BipedalWalkerHardcore 的地形生成主要受 seed 控制。
                # 为了简化，我们假设地形差异在可接受范围内，或者在 teleport 后这一帧迅速调整。
                # *进阶*：如果需要严格一致，需要访问 self.venv.envs[i].seed(case['seed']) 然后 reset。
                # 但这在 VecEnv 中比较 hacky。对于 Retrain 鲁棒性来说，轻微的地形不匹配（Robustness）反而是好事。

                traj = case['trajectory']
                
                # B. 随机倒推时间点
                traj_len = len(traj)
                min_lb, max_lb = self.lookback_window
                safe_max = min(traj_len - 1, max_lb)
                safe_min = min(traj_len - 1, min_lb)
                
                if safe_max > 0:
                    lookback = np.random.randint(safe_min, safe_max + 1)
                    target_idx = max(0, traj_len - lookback)
                    target_state = traj[target_idx]
                    
                    # C. 执行传送
                    # 访问底层的 Gym 环境实例
                    real_env = self.venv.envs[i] 
                    success = teleport_robot(real_env, target_state)
                    
                    if success:
                        # D. 更新 Observation
                        # 机器人位置变了，旧的 obs 已经失效。
                        # 我们通过执行一个空动作来刷新 obs，或者调用 _get_obs
                        # 为了安全起见，我们调用 unwrapped 内部方法（如果存在）
                        # 或者简单地：step 0
                        # 这里的 obs 是 numpy array，我们需要更新它
                        
                        # 尝试获取新 obs
                        unwrapped = real_env.unwrapped
                        if hasattr(unwrapped, '_get_obs'):
                            # BipedalWalker 内部方法
                            new_obs = unwrapped._get_obs()
                            # 处理 VecNormalize 等外层 Wrapper 的影响比较复杂
                            # 简单方案：直接替换 raw obs，虽然可能会导致归一化统计略有偏差，但在 Retrain 中可接受
                            obs[i] = new_obs
                        else:
                            # 备选方案：通过 step(0) 刷新，但这会消耗一步
                            # zero_action = np.zeros(self.action_space.shape)
                            # o, _, _, _ = real_env.step(zero_action)
                            # obs[i] = o
                            pass

        return obs

    def step_wait(self):
        return self.venv.step_wait()


def main():
    parser = argparse.ArgumentParser(description="Retrain agent using Physics Curriculum (State Restoration)")
    parser.add_argument("--env", help="environment ID", type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument("--algo", help="RL Algorithm", default="tqc", type=str, choices=list(ALGOS.keys()))
    parser.add_argument("--model-path", help="Path to the .zip model file to retrain", type=str, required=True)
    # [修改] 参数改为 physics-path
    parser.add_argument("--physics-path", help="Path to the physics_trajectory.pkl file", type=str, required=True)
    
    parser.add_argument("-n", "--n-timesteps", help="Number of timesteps to retrain", default=100000, type=int)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true", default=False, help="Do not render")
    parser.add_argument("--gym-packages", type=str, nargs="+", default=[], help="External Gym packages")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Env constructor kwargs")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs/retrain")
    
    args = parser.parse_args()

    # 1. 导入自定义 Gym 环境
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    # 2. 设置随机种子
    set_random_seed(args.seed)

    # 3. 准备环境配置
    model_dir = os.path.dirname(args.model_path)
    stats_path = os.path.join(model_dir, args.env)
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

    # 4. 创建基础环境 (VecEnv)
    print(f"Creating environment {args.env}...")
    env = create_test_env(
        args.env,
        n_envs=1, 
        stats_path=stats_path, 
        seed=args.seed,
        log_dir=None,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    
    # 恢复 VecNormalize 的训练状态
    if isinstance(env, VecNormalize):
        env.training = True
        env.norm_reward = True 

    # 5. [关键] 应用 PhysicsCurriculumVecWrapper
    if args.physics_path and os.path.exists(args.physics_path):
        print(f"🔥 Applying Physics Curriculum Wrapper from: {args.physics_path}")
        print(f"   -> Probability: 0.2, Lookback: 20~60 steps")
        
        env = PhysicsCurriculumVecWrapper(
            env, 
            physics_file=args.physics_path, 
            sample_prob=0.2,            # 20% 概率进入困难模式
            lookback_window=(20, 60)    # 倒推窗口
        )
    else:
        print(f"Warning: Physics path {args.physics_path} not found! Training will proceed without curriculum.")

    # 6. 加载模型
    print(f"Loading model from {args.model_path}...")
    custom_objects = {}
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    
    # 将包含 Wrapper 的 env 传给 load，这样模型就会在交互时使用这个环境
    model = ALGOS[args.algo].load(args.model_path, env=env, custom_objects=custom_objects)

    # 7. 开始训练
    print(f"Starting retraining for {args.n_timesteps} steps...")
    
    new_log_path = os.path.join(args.folder, f"{args.algo}_retrained_{args.seed}")
    os.makedirs(new_log_path, exist_ok=True)
    
    print(f"Configuring logger to {new_log_path}...")
    new_logger = configure(new_log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    try:
        # 在这里，Reset 由 Wrapper 接管，会自动混入 Crash 状态
        model.learn(total_timesteps=args.n_timesteps)
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    # 8. 保存
    save_name = f"{args.env}_retrained_{args.n_timesteps}_steps"
    save_path = os.path.join(new_log_path, save_name)
    print(f"Saving retrained model to {save_path}.zip")
    model.save(save_path)
    
    if isinstance(env, VecNormalize) or (hasattr(env, 'venv') and isinstance(env.venv, VecNormalize)):
        # 如果最外层是 Wrapper，可能需要剥离找到 VecNormalize
        # 但通常 SB3 的 save 逻辑只能处理它直接持有的 env。
        # 如果我们手动保存，最好尝试保存一下统计数据
        norm_env = env
        while isinstance(norm_env, VecEnvWrapper):
            if isinstance(norm_env, VecNormalize):
                break
            norm_env = norm_env.venv
        
        if isinstance(norm_env, VecNormalize):
            norm_path = os.path.join(new_log_path, f"{args.env}", "vecnormalize.pkl")
            os.makedirs(os.path.dirname(norm_path), exist_ok=True)
            norm_env.save(norm_path)
            print(f"Saved VecNormalize stats to {norm_path}")

    print("Done.")

if __name__ == "__main__":
    main()