import os
# [性能优化] 必须在导入 torch 之前设置，强制单线程以对齐性能
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse, importlib, sys, time, copy, tqdm, pickle, gym, yaml
import numpy as np
import torch as th
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecVideoRecorder, VecNormalize
import utils.import_envs
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
import json, random, math
from datetime import datetime

from interfaces import normalize_data, Memory, Density, compute_sensitivity, case_clip, compute_novelty, Grid
from diffusion import Diffusion

# [性能优化] 限制 PyTorch 内部线程
th.set_num_threads(1)

# =============================================================================
# --- 极速推理接口：绕过 SB3 distribution.py 的高额开销 ---
# =============================================================================
def fast_predict(model, obs):
    """
    直接提取网络均值输出，彻底规避 distribution.py 产生的计算开销。
    """
    obs_tensor = th.as_tensor(obs).float().to("cpu")
    with th.no_grad():
        # 针对 BipedalWalker 常用的 TQC / SAC / TD3 算法
        if hasattr(model.policy, "actor") and hasattr(model.policy.actor, "get_action_dist_params"):
            mean_actions, _, _ = model.policy.actor.get_action_dist_params(obs_tensor)
            # 动作必须经过 tanh 映射到 [-1, 1]
            action = th.tanh(mean_actions).cpu().numpy()
            return action
        # 兜底：如果算法不匹配，使用原版 predict
        return model.predict(obs, deterministic=True)[0]

# ==========================================
# [Alignment] 辅助函数：获取 Raw Observation
# ==========================================
def get_raw_obs(env, obs):
    """
    如果环境被 VecNormalize 包装，则进行反归一化以获取原始物理数值。
    这对于确保 TodyNet 和 Retrain 数据的一致性至关重要。
    """
    norm_env = env
    # 处理嵌套情况 (DummyVecEnv -> VecNormalize)
    if hasattr(norm_env, 'venv') and isinstance(norm_env.venv, VecNormalize):
        return norm_env.venv.unnormalize_obs(obs)
    # 处理直接是 VecNormalize 的情况
    elif isinstance(norm_env, VecNormalize):
        return norm_env.unnormalize_obs(obs)
    # 如果没有归一化，直接返回
    return obs

# ==========================================
# [Helper] 辅助函数：TodyNet 严格采样
# ==========================================
def process_episode_data(sequence, label, window_size):
    seq_len = len(sequence)
    if seq_len < window_size:
        return None, None
    
    seq_array = np.array(sequence) 
    windows = []
    labels = []

    if label == 0:
        # Success: Random 1
        max_idx = seq_len - window_size
        rand_idx = random.randint(0, max_idx)
        win = seq_array[rand_idx : rand_idx + window_size]
        win = win.transpose() 
        windows.append(win)
        labels.append(0)
    else:
        # Failure: Last 1
        win = seq_array[-window_size:] 
        win = win.transpose()
        windows.append(win)
        labels.append(1)
        
    return np.array(windows), np.array(labels)

# ==========================================
# [Helper] 辅助函数：TodyNet 平衡与保存
# ==========================================
def balance_and_save_data(X_list, y_list, output_dir, dataset_name, window_size, target_total=3000, target_crash_ratio=0.30):
    if not X_list:
        return
    
    print(f"\n[TodyNet Data] Processing balancing to {target_total} samples (Target Crash Ratio: {target_crash_ratio:.0%})...")
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    indices_fail = np.where(y_all == 1)[0]
    indices_succ = np.where(y_all == 0)[0]
    
    # 计算目标数量
    target_n_fail = int(target_total * target_crash_ratio)
    target_n_succ = target_total - target_n_fail
    
    print(f"  Raw Collected: Fail={len(indices_fail)}, Success={len(indices_succ)}")
    print(f"  Target: Fail={target_n_fail}, Success={target_n_succ}")

    # 1. 采样 Crash
    if len(indices_fail) >= target_n_fail:
        final_fail = np.random.choice(indices_fail, size=target_n_fail, replace=False)
    else:
        print(f"  [Warning] Not enough crash samples! Keeping all {len(indices_fail)}.")
        final_fail = indices_fail

    # 2. 采样 Success
    if len(indices_succ) >= target_n_succ:
        final_succ = np.random.choice(indices_succ, size=target_n_succ, replace=False)
    else:
        print(f"  [Warning] Not enough success samples! Keeping all {len(indices_succ)}.")
        final_succ = indices_succ
    
    final_indices = np.concatenate([final_fail, final_succ])
    np.random.shuffle(final_indices)
    
    X_balanced = X_all[final_indices]
    y_balanced = y_all[final_indices]

    X_final = np.expand_dims(X_balanced, axis=1)
    X_tensor = th.from_numpy(X_final).float()
    y_tensor = th.from_numpy(y_balanced).long()
    
    total = X_tensor.size(0)
    indices = th.randperm(total)
    split = int(0.8 * total)
    
    ds_id = f"{dataset_name}_{window_size}"
    save_path = os.path.join(output_dir, ds_id)
    os.makedirs(save_path, exist_ok=True)
    
    th.save(X_tensor[indices[:split]], os.path.join(save_path, 'X_train.pt'))
    th.save(y_tensor[indices[:split]], os.path.join(save_path, 'y_train.pt'))
    th.save(X_tensor[indices[split:]], os.path.join(save_path, 'X_valid.pt'))
    th.save(y_tensor[indices[split:]], os.path.join(save_path, 'y_valid.pt'))
    
    final_ratio = y_tensor.float().mean().item()
    print(f"[TodyNet Data] Saved {total} samples to {save_path} | Final Crash Ratio: {final_ratio:.2%}")

# --- [Helper] 辅助函数：获取底层环境以访问 hull 数据 ---
def get_real_unwrapped_env(env):
    current_env = env
    while hasattr(current_env, 'venv'):
        current_env = current_env.venv
    if hasattr(current_env, 'envs'):
        return current_env.envs[0].unwrapped
    return current_env.unwrapped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="../rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="tqc", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=300, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument("--no-render", action="store_true", default=False, help="Do not render the environment")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--load-best", action="store_true", default=False, help="Load best model")
    parser.add_argument("--load-checkpoint", type=int, help="Load checkpoint")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--norm-reward", action="store_true", default=False, help="Normalize reward")
    parser.add_argument("--vecnormalize-path", default=None, help="Optional path to vecnormalize.pkl")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument("--gym-packages", type=str, nargs="+", default=[], help="External Gym packages")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Env constructor kwargs")

    # Generative testing parameters
    parser.add_argument("--method", help="select the guidance for testing", default="generative", type=str, required=False)
    parser.add_argument("--hour", help="test time", default=12, type=int)
    parser.add_argument("--step", help="number of normal cases at each training step", default=50, type=int)
    parser.add_argument("--grid", help="state abstraction granularity", default=5, type=int)
    
    # [Alignment] 数据采集参数
    parser.add_argument("--save-data", action="store_true", default=False, help="Save TodyNet training data")
    parser.add_argument("--save-transitions", action="store_true", default=False, help="Save RL transitions")
    parser.add_argument("--window-size", type=int, default=20, help="Sliding window size")
    parser.add_argument("--dataset-name", type=str, default="BipedalWalkerHC", help="Dataset name prefix")
    
    args = parser.parse_args()
    
    result_folder_name = f"{args.method}_{args.step}_seed_{args.seed}"
    result_path = os.path.join('results', result_folder_name)
    os.makedirs(result_path, exist_ok=True)
    log_file_path = os.path.join(result_path, 'run_log.txt')
    f = open(log_file_path, 'w', buffering=1)
    sys.stdout = f
    sys.stderr = f

    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)

    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    # 强制线程数为 1 (即使 argparse 传入了其他值，这里也由全局优化决定)
    th.set_num_threads(1)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    if args.vecnormalize_path is not None:
        stats_path = os.path.dirname(args.vecnormalize_path)
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

    log_dir = args.reward_log if args.reward_log != "" else None

    # 创建环境
    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        kwargs.update(dict(buffer_size=1))

    custom_objects = {}
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    # [优化] 强制加载模型到 CPU
    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, device="cpu", **kwargs)

    # Diffusion Setup
    case_dimension = 15
    diffusion_model = Diffusion(batch_size = 1, epoch = 100, data_size = case_dimension, training_step_per_spoch = 25, num_diffusion_step = 25)
    diffusion_model.setup()
    memory_model = Memory(size = 100)
    density_model = Density()

    # Novelty Grid Setup
    min_obs = np.array([-5 for i in range(env.observation_space.shape[0])])
    max_obs = np.array([5 for i in range(env.observation_space.shape[0])])
    novelty_grid = Grid(min_obs, max_obs, args.grid)
    novelty_test_grid = Grid(min_obs, max_obs, args.grid)
    novelty_dict = dict()
    novelty_test_dict = dict()

    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    # Init variables
    normal_case_list = []
    metric_list = []
    density_list = []
    sensitivity_list = []
    performance_list = []
    diffusion_failure_list = []
    diffusion_failure_clusters = []
    random_failure_list = []
    diffusion_failure_count = []
    information_list = []
    
    # [Alignment] 数据采集容器初始化
    all_window_data = [] 
    all_label_data = []
    todynet_success_count = 0  
    TODYNET_SUCCESS_SOFT_CAP = 5000
    
    crash_transitions = []
    success_transitions = []
    TARGET_CRASH_COUNT = 10000
    TARGET_SUCCESS_COUNT = 20000
    all_test_cases_log = []

    # --- Stage 1: Initialization (Warm-up) ---
    print("--- Stage 1: Initialization (Warm-up) ---")
    initial_collection_count = 1000
    for pre_step in tqdm.tqdm(range(initial_collection_count), desc="Initial Random Sampling"):
        normal_case = np.random.randint(low=1, high=4, size=15)
        normal_case_list.append(normal_case)

    # --- Stage 2: Pre-training Diffusion ---
    if len(normal_case_list) > 0:
        print(f"--- Pre-training Diffusion Model with {len(normal_case_list)} samples ---")
        normal_case_list = np.array(normal_case_list)
        diffusion_model.train(normal_case_list, None, 'generative')
        normal_case_list = []
        metric_list = []
        memory_model.clear()
        
    # --- Stage 3: Main Testing Loop ---
    start_time = time.time()
    current_time = time.time()
    cur_step = 0
    wins = 0
    lose = 0
    failure_by_diffusion = 0

    # [新增] 对齐评估指标：初始化数据结构和时间统计容器
    selection_log_data = []
    total_env_sim_time = 0.0
    fuzzing_start_time = start_time

    print("--- Stage 2: Main Testing Loop ---")
    while current_time - start_time < 3600 * 12: 

        if cur_step > 0 and cur_step % args.step == 0:
            # --- Generative Phase ---
            normal_case_list = np.array(normal_case_list)
            metric_list      = np.array(metric_list)

            if args.method == 'generative': metrics = None
            elif args.method == 'generative+density': metrics = metric_list[:, [0]]
            elif args.method == 'generative+sensitivity': metrics = metric_list[:, [1]]
            elif args.method == 'generative+performance': metrics = metric_list[:, [2]]
            elif args.method == 'generative+baseline': metrics = metric_list[:, [0,1,2]]
            elif args.method == 'generative+novelty': metrics = metric_list[:, [3]]
            else:
                print('Please check the method parameters!')
                return

            diffusion_model.train(normal_case_list, metrics, args.method)
            normal_case_list = []
            metric_list = []
            memory_model.clear()

            for _ in range(50): # val_step fixed to 50
                failure_flag = False
                state = None
                test_case = diffusion_model.generate()
                
                # [新增] 对齐评估指标：记录环境重置时间开销
                t0_env = time.time()
                obs = env.reset(test_case)
                total_env_sim_time += (time.time() - t0_env)

                sequences = [obs[0]]
                episode_reward = 0.0
                
                todynet_sequences = []   
                current_ep_transitions = []
                
                total_x_pos_sum = 0.0
                total_abs_angle_sum = 0.0
                episode_steps = 0

                for _ in range(args.n_timesteps):
                    # [Alignment] 获取 Raw Observation (当前步)
                    current_obs_norm = obs[0].copy()
                    current_obs_raw = get_raw_obs(env, current_obs_norm)

                    # [替换] 使用极速接口规避分布开销
                    action = fast_predict(model, obs)
                    current_action = action[0]

                    # [Alignment] TodyNet 使用 Raw Data
                    vec_28d = np.concatenate((current_obs_raw, current_action))
                    todynet_sequences.append(vec_28d)

                    # 执行动作
                    # [新增] 对齐评估指标：记录物理仿真步进时间开销
                    t0_env = time.time()
                    obs, reward, done, infos = env.step(action)
                    total_env_sim_time += (time.time() - t0_env)
                    
                    # [Alignment] 获取 Raw Observation (下一步)
                    next_obs_norm = obs[0].copy()
                    next_obs_raw = get_raw_obs(env, next_obs_norm)

                    # [Alignment] Transitions 使用 Raw Data
                    if args.save_transitions:
                        # 格式: (Raw_Obs, Action, Reward, Raw_Next_Obs, Done)
                        current_ep_transitions.append((current_obs_raw, current_action, reward[0], next_obs_raw, done[0]))
                    
                    # Diversity 计算
                    real_env = get_real_unwrapped_env(env)
                    if hasattr(real_env, 'hull'):
                        total_x_pos_sum += real_env.hull.position[0]
                        total_abs_angle_sum += abs(real_env.hull.angle)
                        episode_steps += 1
                    
                    sequences.append(obs[0])
                    episode_reward += reward[0]
                    if done:
                        break
                
                bd_dist = total_x_pos_sum / max(1, episode_steps)
                bd_mean_angle = total_abs_angle_sum / max(1, episode_steps)
                is_crash = (done or episode_reward < 10)
                elapsed_time = time.time() - start_time

                # [新增] 对齐评估指标：精细拆解失败状态
                actual_done = bool(done[0] if isinstance(done, (list, np.ndarray)) else done)
                did_crash_flag = actual_done
                is_reward_fault_flag = (not actual_done) and (episode_reward < 10)
                
                selection_log_data.append({
                    'mutate_state':np.array(test_case, dtype=np.int32),
                    'did_crash': did_crash_flag,
                    'is_reward_fault': is_reward_fault_flag,
                    'elapsed_time': time.time() - fuzzing_start_time,
                    'survival_steps': episode_steps,
                    'parent_depth': 0,
                    'output_trajectory': np.array(sequences, dtype=np.float32) if (did_crash_flag or is_reward_fault_flag) else None
                })
                
                # [Data Collection] 
                if args.save_transitions:
                    if is_crash:
                        if len(crash_transitions) < TARGET_CRASH_COUNT:
                            crash_transitions.extend(current_ep_transitions)
                    else:
                        if len(success_transitions) < TARGET_SUCCESS_COUNT:
                            success_transitions.extend(current_ep_transitions)

                if args.save_data:
                    label = 1 if is_crash else 0
                    collect_this = (label == 1) or (todynet_success_count < TODYNET_SUCCESS_SOFT_CAP)
                    if collect_this:
                        wins_data, labels_data = process_episode_data(todynet_sequences, label, args.window_size)
                        if wins_data is not None and len(wins_data) > 0:
                            all_window_data.append(wins_data)
                            all_label_data.append(labels_data)
                            if label == 0: todynet_success_count += 1
                
                all_test_cases_log.append({
                    "input": test_case.tolist(), "is_crash": bool(is_crash), "source": "generative",
                    "step": cur_step, "time": elapsed_time, "bd_distance": bd_dist, "bd_mean_angle": bd_mean_angle 
                })

                if is_crash:
                    save_case = test_case.tolist()
                    if save_case not in random_failure_list and save_case not in diffusion_failure_list:
                        failure_flag = True
                        lose += 1
                        diffusion_failure_list.append(save_case)
                        failure_by_diffusion += 1
                        print(f"{(current_time - start_time)/3600:.2f}h | Crash | {save_case}")
                        diffusion_failure_count.append([(current_time - start_time)/3600, failure_by_diffusion, save_case])
                else:
                    wins += 1  

                abstract_id = novelty_grid.state_abstract(np.array([sequences[-1]]))[0]
                novelty_dict[abstract_id] = novelty_dict.get(abstract_id, 0) + 1
                novelty = novelty_dict[abstract_id]
                
                norm_novelty = math.exp(-(novelty - 1)) 

                normal_case_list.append(test_case)
                metric_list.append([0, 0, 0, norm_novelty])
                memory_model.append(test_case, 0, 0, 0, novelty)

                if failure_flag:
                    diffusion_failure_clusters.append(abstract_id)
                information_list.append([sequences[-1].tolist(), failure_flag, abstract_id, norm_novelty])
        else:
            # --- Random Sampling Phase ---
            state = None
            normal_case = np.random.randint(low=1, high=4, size=15)
            
            # [新增] 对齐评估指标：记录环境重置时间开销
            t0_env = time.time()
            obs = env.reset(normal_case)
            total_env_sim_time += (time.time() - t0_env)

            sequences = [obs[0]]
            episode_reward = 0.0
            
            total_x_pos_sum = 0.0
            total_abs_angle_sum = 0.0
            episode_steps = 0
            
            for _ in range(args.n_timesteps):
                # [替换] 使用极速接口规避分布开销
                action = fast_predict(model, obs)
                
                # [新增] 对齐评估指标：记录物理仿真步进时间开销
                t0_env = time.time()
                obs, reward, done, infos = env.step(action)
                total_env_sim_time += (time.time() - t0_env)

                real_env = get_real_unwrapped_env(env)
                if hasattr(real_env, 'hull'):
                    total_x_pos_sum += real_env.hull.position[0]
                    total_abs_angle_sum += abs(real_env.hull.angle)
                    episode_steps += 1
                
                sequences.append(obs[0])
                episode_reward += reward[0]
                if done: break
            
            bd_dist = total_x_pos_sum / max(1, episode_steps)
            bd_mean_angle = total_abs_angle_sum / max(1, episode_steps)
            is_crash = (done or episode_reward < 10)
            elapsed_time = time.time() - start_time

            # [新增] 对齐评估指标：精细拆解失败状态
            actual_done = bool(done[0] if isinstance(done, (list, np.ndarray)) else done)
            did_crash_flag = actual_done
            is_reward_fault_flag = (not actual_done) and (episode_reward < 10)
            
            selection_log_data.append({
                'mutate_state': np.array(normal_case, dtype=np.int32),
                'did_crash': did_crash_flag,
                'is_reward_fault': is_reward_fault_flag,
                'elapsed_time': time.time() - fuzzing_start_time,
                'survival_steps': episode_steps,
                'parent_depth': 0,
                'output_trajectory': np.array(sequences, dtype=np.float32) if (did_crash_flag or is_reward_fault_flag) else None
            })
            
            all_test_cases_log.append({
                "input": normal_case.tolist(), "is_crash": bool(is_crash), "source": "random",
                "step": cur_step, "time": elapsed_time, "bd_distance": bd_dist, "bd_mean_angle": bd_mean_angle 
            })

            normal_case_list.append(normal_case)
            
            # Metrics calculation
            density, norm_density = 0, 0
            sensitivity, norm_sensitivity = 0, 0
            performance, norm_performance = 0, 0
            novelty, norm_novelty = 0, 0
            cases_list = memory_model.get_cases()

            if 'density' in args.method:
                density_list = memory_model.get_densities()
                density = density_model.state_coverage(sequences)
                norm_density = normalize_data(density, memory_model.min_density, memory_model.max_density)
            
            if 'sensitivity' in args.method:
                sensitivity_list = memory_model.get_sensitivities()
                sensitivity = compute_sensitivity(normal_case, cases_list, performance_list, episode_reward)
                norm_sensitivity = normalize_data(sensitivity, memory_model.min_sensitivity, memory_model.max_sensitivity)
                norm_sensitivity = 1 - norm_sensitivity
            
            if 'performance' in args.method:
                performance_list = memory_model.get_performances()
                performance = episode_reward
                norm_performance = normalize_data(performance, memory_model.min_performance, memory_model.max_performance)
            
            if 'novelty' in args.method:
                abstract_id = novelty_grid.state_abstract(np.array([sequences[-1]]))[0]
                novelty_dict[abstract_id] = novelty_dict.get(abstract_id, 0) + 1
                novelty = novelty_dict[abstract_id]
                norm_novelty = math.exp(-(novelty - 1))

            metric_list.append([norm_density, norm_sensitivity, norm_performance, norm_novelty])
            memory_model.append(normal_case, density, sensitivity, performance, novelty)
            print(f"Step {cur_step} | Random Case")

        cur_step += 1
        current_time = time.time()

    # ==========================================
    # [Alignment] 保存阶段: 结构对齐
    # ==========================================
    
    # 1. 保存 Transitions (Dict format, Raw Data)
    if args.save_transitions:
        save_payload = {
            "crash": crash_transitions,
            "success": success_transitions,
            "is_raw": True
        }
        trans_file = os.path.join(result_path, 'transitions.pkl')
        print(f"Saving labeled RAW transitions (Dict) to {trans_file}...")
        with open(trans_file, 'wb') as f_t:
            pickle.dump(save_payload, f_t, protocol=pickle.HIGHEST_PROTOCOL)
        print("Transitions saved.")

    # 2. 保存 TodyNet 数据 (Raw Tensors)
    if args.save_data:
        balance_and_save_data(
            all_window_data, 
            all_label_data, 
            result_path, 
            args.dataset_name, 
            args.window_size,
            target_total=3000,
            target_crash_ratio=0.30
        )

    # Save other logs
    with open(os.path.join(result_path, 'diffusion_failure_count.json'), 'w') as f:
        json.dump(diffusion_failure_count, f)
    with open(os.path.join(result_path, 'information.json'), 'w') as f:
        json.dump(information_list, f)
    with open(os.path.join(result_path, 'novelty_dict.json'), 'w') as f:
        json.dump(novelty_dict, f)
    with open(os.path.join(result_path, 'all_test_cases_log.pkl'), 'wb') as f:
        pickle.dump(all_test_cases_log, f)

    # [新增] 对齐评估指标：将核心数据及性能元数据严格按照要求写入文件
    selection_log_path = os.path.join(result_path, 'selection_log.pkl')
    with open(selection_log_path, 'wb') as f_sel:
        pickle.dump(selection_log_data, f_sel, protocol=pickle.HIGHEST_PROTOCOL)
        
    total_wall_time_val = time.time() - fuzzing_start_time
    perf_meta_data = {
        'total_wall_time': total_wall_time_val,
        'env_sim_time': total_env_sim_time,
        'algo_logic_time': total_wall_time_val - total_env_sim_time
    }
    perf_meta_path = os.path.join(result_path, 'perf_meta.pkl')
    with open(perf_meta_path, 'wb') as f_perf:
        pickle.dump(perf_meta_data, f_perf, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':  
    start_time = datetime.now()
    print(f"--- start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    main()
    end_time = datetime.now()
    print(f"--- finish time: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"--- total time: {end_time - start_time} ---")
