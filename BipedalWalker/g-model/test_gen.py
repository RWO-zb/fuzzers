import argparse, importlib, os, sys, time, copy, tqdm, pickle, gym, yaml
import numpy as np
import torch as th
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecVideoRecorder
import utils.import_envs
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
import json, random, math
from datetime import datetime

from interfaces import normalize_data, Memory, Density, compute_sensitivity, case_clip, compute_novelty, Grid
from diffusion import Diffusion

# ==========================================
# [新增] 辅助函数：TodyNet 严格采样
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
# [新增] 辅助函数：TodyNet 平衡与保存 (3000条, 30% Crash)
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

# --- [新增] 辅助函数：获取底层环境以访问 hull 数据 ---
def get_real_unwrapped_env(env):
    """
    穿透 VecEnv 和 Monitor 等包装器，获取底层的 BipedalWalker 环境实例，
    以便访问 hull.position 和 hull.angle。
    """
    current_env = env
    while hasattr(current_env, 'venv'):
        current_env = current_env.venv
    
    if hasattr(current_env, 'envs'):
        # 对于 DummyVecEnv，通常取第一个环境
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

    ######################## parameters for generative testing ############################################
    parser.add_argument("--method", help="select the guidance for testing", default="generative", type=str, required=False)
    parser.add_argument("--hour", help="test time", default=12, type=int)
    parser.add_argument("--step", help="number of normal cases at each training step", default=50, type=int)
    parser.add_argument("--grid", help="state abstraction granularity", default=5, type=int)
    
    # [新增] 数据采集相关参数
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

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)

    # Sanity checks
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

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

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

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    ##################################################################################################

    case_dimension = 15
    diffusion_model = Diffusion(batch_size = 1, epoch = 100, data_size = case_dimension, training_step_per_spoch = 25, num_diffusion_step = 25)
    diffusion_model.setup()
    memory_model = Memory(size = 100)
    density_model = Density()


    ################################### nvovelty computation ########################################
    min_obs = np.array([-5 for i in range(env.observation_space.shape[0])])
    max_obs = np.array([5 for i in range(env.observation_space.shape[0])])
    novelty_grid = Grid(min_obs, max_obs, args.grid)
    novelty_test_grid = Grid(min_obs, max_obs, args.grid)
    novelty_dict = dict()
    novelty_test_dict = dict()



    #np.random.seed()
    states = np.random.randint(low=1, high=4, size=15)
    obs = env.reset(states)

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    episode_rewards, episode_lengths = [], []
    ep_len = 0

    total_step = 100000
    val_step = 50
    cur_step = 0
    wins = 0
    lose = 0
    failure_by_diffusion = 0
    failure_by_random = 0
    done = False
    regular_time = 0
    normal_case_list = []
    metric_list = []
    density_list = []

    sensitivity_list = []
    performance_list = []
    novelty_list = []
    diffusion_failure_list = []
    diffusion_failure_clusters = []
    random_failure_list = []
    diffusion_failure_count = []
    random_failure_count = []
    #######################################################################################
    trajectory_list = []
    termination_list = []
    information_list = []
    failure_flag = False
    
    # ==========================================
    # [新增] 数据采集容器初始化
    # ==========================================
    # TodyNet 容器
    all_window_data = [] 
    all_label_data = []
    todynet_success_count = 0  
    TODYNET_SUCCESS_SOFT_CAP = 5000
    
    # Transitions 容器
    crash_transitions = []
    success_transitions = []
    TARGET_CRASH_COUNT = 10000
    TARGET_SUCCESS_COUNT = 20000

    # --- 阶段 1：严格遵循论文的初始化预热 (Strict Initialization) ---
    print("--- Stage 1: Initialization (Warm-up) ---")
    initial_collection_count = 1000
    
    for pre_step in tqdm.tqdm(range(initial_collection_count), desc="Initial Random Sampling"):
        # 仅生成随机的初始状态，不进行 env.step() 执行
        normal_case = np.random.randint(low=1, high=4, size=15)
        normal_case_list.append(normal_case)

    # --- 阶段 2：预训练扩散模型 (Pre-training) ---
    if len(normal_case_list) > 0:
        print(f"--- Pre-training Diffusion Model with {len(normal_case_list)} samples ---")
        normal_case_list = np.array(normal_case_list)
        
        diffusion_model.train(normal_case_list, None, 'generative')
        
        normal_case_list = []
        metric_list = []
        memory_model.clear()
        
    # --- 阶段 3：正式测试循环 (Main Loop) ---
    start_time = time.time()
    current_time = time.time()
    
    # 初始化日志列表
    all_test_cases_log = []

    print("--- Stage 2: Main Testing Loop ---")
    while current_time - start_time < 3600 * 4 : # 使用参数控制时间

        if cur_step > 0 and cur_step % args.step == 0:
            # --- 扩散模型微调与生成阶段 ---
            normal_case_list = np.array(normal_case_list)
            metric_list      = np.array(metric_list)

            if args.method == 'generative':
                metrics = None
            elif args.method == 'generative+density':
                metrics = metric_list[:, [0]]
            elif args.method == 'generative+sensitivity':
                metrics = metric_list[:, [1]]
            elif args.method == 'generative+performance':
                metrics = metric_list[:, [2]]
            elif args.method == 'generative+baseline':
                metrics = metric_list[:, [0,1,2]]
            elif args.method == 'generative+novelty':
                metrics = metric_list[:, [3]]
            else:
                print('Please check the method parameters!')
                return

            diffusion_model.train(normal_case_list, metrics, args.method)
            normal_case_list = []
            metric_list = []
            memory_model.clear()

            for _ in range(val_step):
                failure_flag = False
                state = None
                test_case = diffusion_model.generate()
                obs = env.reset(test_case)
                sequences = [obs[0]]
                episode_reward = 0.0
                
                # [新增] 采集变量
                todynet_sequences = []   
                current_ep_transitions = []
                
                # [新增] 行为多样性统计变量
                total_x_pos_sum = 0.0
                total_abs_angle_sum = 0.0
                episode_steps = 0

                for _ in range(args.n_timesteps):
                    current_obs = obs[0].copy() # 捕获当前状态
                    action, state = model.predict(obs, state=state, deterministic=deterministic)
                    current_action = action[0]  # 捕获当前动作
                    
                    # TodyNet 向量
                    vec_28d = np.concatenate((current_obs, current_action))
                    todynet_sequences.append(vec_28d)

                    obs, reward, done, infos = env.step(action)
                    
                    # Transitions 记录
                    if args.save_transitions:
                        current_ep_transitions.append((current_obs, current_action, reward[0], obs[0].copy(), done[0]))
                    
                    # [新增] 获取底层数据以计算 Diversity
                    real_env = get_real_unwrapped_env(env)
                    if hasattr(real_env, 'hull'):
                        raw_x_pos = real_env.hull.position[0]
                        raw_angle = real_env.hull.angle
                        total_x_pos_sum += raw_x_pos
                        total_abs_angle_sum += abs(raw_angle)
                        episode_steps += 1
                    
                    sequences.append(obs[0])
                    episode_reward += reward[0]
                    if done:
                        break
                
                # [新增] 计算行为特征
                bd_dist = total_x_pos_sum / max(1, episode_steps)
                bd_mean_angle = total_abs_angle_sum / max(1, episode_steps)

                is_crash = (done or episode_reward < 10)
                elapsed_time = time.time() - start_time
                
                # === [新增] 数据采集逻辑 (仅在 Generative 阶段) ===
                # 1. Transitions 收集
                if args.save_transitions:
                    if is_crash:
                        if len(crash_transitions) < TARGET_CRASH_COUNT:
                            crash_transitions.extend(current_ep_transitions)
                    else:
                        if len(success_transitions) < TARGET_SUCCESS_COUNT:
                            success_transitions.extend(current_ep_transitions)

                # 2. TodyNet 数据收集
                if args.save_data:
                    label = 1 if is_crash else 0
                    collect_this = False
                    if label == 1:
                        collect_this = True
                    else:
                        if todynet_success_count < TODYNET_SUCCESS_SOFT_CAP:
                            collect_this = True
                    
                    if collect_this:
                        wins_data, labels_data = process_episode_data(todynet_sequences, label, args.window_size)
                        if wins_data is not None and len(wins_data) > 0:
                            all_window_data.append(wins_data)
                            all_label_data.append(labels_data)
                            if label == 0:
                                todynet_success_count += 1
                
                # [新增] 记录 bd_distance 和 bd_mean_angle
                all_test_cases_log.append({
                    "input": test_case.tolist(), 
                    "is_crash": bool(is_crash),
                    "source": "generative",
                    "step": cur_step,
                    "time": elapsed_time,
                    "bd_distance": bd_dist,      
                    "bd_mean_angle": bd_mean_angle 
                })

                if is_crash:
                    save_case = test_case.tolist()
                    if save_case in random_failure_list or save_case in diffusion_failure_list:
                        pass
                    else:
                        failure_flag = True
                        lose += 1
                        done = False 
                        regular_time = (current_time - start_time) / 3600
                        diffusion_failure_list.append(save_case)
                        failure_by_diffusion += 1
                        print(regular_time, failure_by_diffusion, save_case)
                        diffusion_failure_count.append([regular_time, failure_by_diffusion, save_case])
                else:
                    wins += 1  

                abstract_id = novelty_grid.state_abstract(np.array([sequences[-1]]))[0]
                if abstract_id in novelty_test_dict.keys():
                    novelty_dict[abstract_id] += 1
                else:
                    novelty_dict[abstract_id] = 1
                novelty = novelty_dict[abstract_id]
                norm_novelty = 1 / (math.e ** (novelty - 1))

                normal_case_list.append(test_case)
                metric_list.append([0, 0, 0, norm_novelty])
                memory_model.append(test_case, 0, 0, 0, novelty)

                if failure_flag:
                    diffusion_failure_clusters.append(abstract_id)

                print(failure_flag, abstract_id, len(novelty_dict.keys()), len(set(diffusion_failure_clusters)))
                information_list.append([sequences[-1].tolist(), failure_flag, abstract_id, norm_novelty])
        else:
            # --- 随机采样阶段 (不采集数据) ---
            state = None
            normal_case = np.random.randint(low=1, high=4, size=15)
            
            obs = env.reset(normal_case)
            sequences = [obs[0]]
            episode_reward = 0.0
            
            # [新增] 行为多样性统计变量
            total_x_pos_sum = 0.0
            total_abs_angle_sum = 0.0
            episode_steps = 0
            
            for _ in range(args.n_timesteps):
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                
                # [新增] 获取底层数据以计算 Diversity
                real_env = get_real_unwrapped_env(env)
                if hasattr(real_env, 'hull'):
                    raw_x_pos = real_env.hull.position[0]
                    raw_angle = real_env.hull.angle
                    total_x_pos_sum += raw_x_pos
                    total_abs_angle_sum += abs(raw_angle)
                    episode_steps += 1
                
                sequences.append(obs[0])
                episode_reward += reward[0]
                if done:
                    break
            
            # [新增] 计算行为特征
            bd_dist = total_x_pos_sum / max(1, episode_steps)
            bd_mean_angle = total_abs_angle_sum / max(1, episode_steps)

            is_crash = (done or episode_reward < 10)
            elapsed_time = time.time() - start_time
            
            # [新增] 记录 bd_distance 和 bd_mean_angle
            all_test_cases_log.append({
                "input": normal_case.tolist(), 
                "is_crash": bool(is_crash),
                "source": "random",
                "step": cur_step,
                "time": elapsed_time,
                "bd_distance": bd_dist,      
                "bd_mean_angle": bd_mean_angle 
            })

            normal_case_list.append(normal_case)
            
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
                metric = norm_sensitivity
            
            if 'performance' in args.method:
                performance_list = memory_model.get_performances()
                performance = episode_reward
                norm_performance = normalize_data(performance, memory_model.min_performance, memory_model.max_performance)
            
            if 'novelty' in  args.method:
                abstract_id = novelty_grid.state_abstract(np.array([sequences[-1]]))[0]
                if abstract_id in novelty_dict.keys():
                    novelty_dict[abstract_id] += 1
                else:
                    novelty_dict[abstract_id] = 1
                novelty = novelty_dict[abstract_id]
                norm_novelty = 1 / (math.e ** (novelty - 1))

            metric_list.append([norm_density, norm_sensitivity, norm_performance, norm_novelty])
            memory_model.append(normal_case, density, sensitivity, performance, novelty)

            print(cur_step, normal_case)

        cur_step += 1
        current_time = time.time()

    # ==========================================
    # [新增] 循环结束后的数据保存逻辑
    # ==========================================
    
    # 1. 保存 Transitions
    if args.save_transitions:
        final_transitions = crash_transitions + success_transitions
        random.shuffle(final_transitions)
        
        trans_file = os.path.join(result_path, 'transitions.pkl')
        print(f"Saving {len(final_transitions)} total transitions to {trans_file}...")
        with open(trans_file, 'wb') as f_t:
            pickle.dump(final_transitions, f_t, protocol=pickle.HIGHEST_PROTOCOL)
        print("Transitions saved.")

    # 2. 保存 TodyNet 数据
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

    file_path_diffusion = os.path.join(result_path, 'diffusion_failure_count.json')
    with open(file_path_diffusion, 'w') as f:
        json.dump(diffusion_failure_count, f)

    file_path_info = os.path.join(result_path, 'information.json')
    with open(file_path_info, 'w') as f:
        json.dump(information_list, f)

    file_path_novelty = os.path.join(result_path, 'novelty_dict.json')
    with open(file_path_novelty, 'w') as f:
        json.dump(novelty_dict, f)
        
    log_filename = os.path.join(result_path, 'all_test_cases_log.pkl')
    with open(log_filename, 'wb') as f:
        pickle.dump(all_test_cases_log, f)

if __name__ == '__main__':  
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- start time: {start_time_str} ---")
    main()
    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- finish time: {end_time_str} ---")
    duration = end_time - start_time
    print(f"--- total time: {duration} ---")