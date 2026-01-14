import argparse
import importlib
import os
import sys
import time
import copy
import tqdm
import pickle
import yaml
import numpy as np
import gym
import torch
import random
from datetime import datetime
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from fuzz.cure_fuzz import CureFuzz

# ==========================================
# [新增] 物理状态提取辅助函数
# ==========================================
def extract_physics_state(real_env):
    """
    从底层 BipedalWalker 环境中提取完整的物理状态 (Box2D)
    """
    # 确保拿到的是 BipedalWalker 类实例
    if not hasattr(real_env, 'hull') or not hasattr(real_env, 'legs'):
        return None

    hull = real_env.hull
    legs = real_env.legs # 列表 [leg1_upper, leg1_lower, leg2_upper, leg2_lower]
    
    state_dict = {
        # 1. 躯干 (Hull)
        "hull_pos": (hull.position[0], hull.position[1]),
        "hull_angle": hull.angle,
        "hull_lin_vel": (hull.linearVelocity[0], hull.linearVelocity[1]),
        "hull_ang_vel": hull.angularVelocity,
        
        # 2. 腿部 (Legs)
        "legs": []
    }
    
    for leg_part in legs:
        leg_data = {
            "pos": (leg_part.position[0], leg_part.position[1]),
            "angle": leg_part.angle,
            "lin_vel": (leg_part.linearVelocity[0], leg_part.linearVelocity[1]),
            "ang_vel": leg_part.angularVelocity,
        }
        state_dict["legs"].append(leg_data)
        
    return state_dict

# ==========================================
# [辅助函数] TodyNet 严格采样
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
# [辅助函数] TodyNet 平衡与截断
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
    X_tensor = torch.from_numpy(X_final).float()
    y_tensor = torch.from_numpy(y_balanced).long()
    
    total = X_tensor.size(0)
    indices = torch.randperm(total)
    split = int(0.8 * total)
    
    ds_id = f"{dataset_name}_{window_size}"
    save_path = os.path.join(output_dir, ds_id)
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(X_tensor[indices[:split]], os.path.join(save_path, 'X_train.pt'))
    torch.save(y_tensor[indices[:split]], os.path.join(save_path, 'y_train.pt'))
    torch.save(X_tensor[indices[split:]], os.path.join(save_path, 'X_valid.pt'))
    torch.save(y_tensor[indices[split:]], os.path.join(save_path, 'y_valid.pt'))
    
    final_ratio = y_tensor.float().mean().item()
    print(f"[TodyNet Data] Saved {total} samples to {save_path} | Final Crash Ratio: {final_ratio:.2%}")

def get_real_unwrapped_env(env):
    current_env = env
    while hasattr(current_env, 'venv'):
        current_env = current_env.venv
    if hasattr(current_env, 'envs'):
        return current_env.envs[0].unwrapped
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="../rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="tqc", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=300, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode", default=1, type=int)
    parser.add_argument("--no-render", action="store_true", default=False, help="Do not render")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--load-best", action="store_true", default=False, help="Load best model")
    parser.add_argument("--load-checkpoint", type=int, help="Load specific checkpoint")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--norm-reward", action="store_true", default=False, help="Normalize reward")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument("--gym-packages", type=str, nargs="+", default=[], help="External Gym packages")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Env constructor kwargs")
    parser.add_argument("--guide", action="store_true", default=False)
    parser.add_argument("--intrinsic", help="Threshold for intrinsic reward", default=10, type=int)
    parser.add_argument("--entropy", help="Threshold for reward", default=10, type=int)
    parser.add_argument("--seed_number", help="Number of seeds", default=100, type=int)
    
    parser.add_argument("--save-data", action="store_true", default=False, help="Save TodyNet training data")
    parser.add_argument("--save-transitions", action="store_true", default=False, help="Save RL transitions")
    # [新增参数] 开启物理状态收集
    parser.add_argument("--save-physics", action="store_true", default=False, help="Save full physics state trajectories for crash restoration")
    
    parser.add_argument("--window-size", type=int, default=25, help="Sliding window size")
    parser.add_argument("--dataset-name", type=str, default="BipedalWalkerHC", help="Dataset name prefix")

    args = parser.parse_args()
    
    now_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    result_folder = f"{now_str}_seed_{args.seed}"
    result_path = './results/' + result_folder + '/'
    os.makedirs(result_path, exist_ok=True)
    
    log_file_path = os.path.join(result_path, 'cure_fuzz.txt')
    f = open(log_file_path, 'w', buffering=1, encoding='utf-8')
    sys.stdout = f
    sys.stderr = f

    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    intrins_theta = args.intrinsic
    entropy_theta = args.entropy
    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)

    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    found = False
    model_path = ""
    for ext in ["zip"]:
        path = os.path.join(log_path, f"{env_id}.{ext}")
        if os.path.isfile(path):
            model_path = path
            found = True
            break

    if args.load_best:
        path = os.path.join(log_path, "best_model.zip")
        if os.path.isfile(path):
            model_path = path
            found = True

    if args.load_checkpoint is not None:
        path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        if os.path.isfile(path):
            model_path = path
            found = True

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)
    is_atari = ExperimentManager.is_atari(env_id)
    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f_args:
            loaded_args = yaml.load(f_args, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=args.reward_log if args.reward_log != "" else None,
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

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
    
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic
    fuzzer = CureFuzz()
    seeds_num = args.seed_number
    
    # === TodyNet 容器 ===
    all_window_data = [] 
    all_label_data = []
    todynet_success_count = 0 
    
    # === Transitions 容器 ===
    crash_transitions = []
    success_transitions = []
    
    # === [新增] 物理轨迹容器 ===
    # 结构: [{'seed': int, 'trajectory': [dict, dict, ...]}, ...]
    all_crash_physics_trajectories = []
    
    # === [配置] 采集目标 ===
    TARGET_CRASH_COUNT = 10000
    TARGET_SUCCESS_COUNT = 20000
    TODYNET_SUCCESS_SOFT_CAP = 5000

    pbar = tqdm.tqdm(total=seeds_num)
    start_corpus_time = time.time()
    i = 0
    
    # --- Corpus Generation ---
    while i < seeds_num and (time.time() - start_corpus_time) <= (3600*2):
        states = np.random.randint(low=1, high=4, size=15)
        state = None
        episode_reward = 0.0
        obs = env.reset(states)
        sequences = [obs[0]] 
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            sequences.append(obs[0])
            episode_reward += reward[0]
            if done:
                break
        final_state = sequences[-2]
        
        state = None
        delta_states = np.random.choice(2, 15, p=[0.9, 0.1])
        if np.sum(delta_states) == 0:
            delta_states[0] = 1
        mutate_states = np.clip(np.remainder(states + delta_states, 4), 1, 3)

        obs = env.reset(mutate_states)
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, _, done, _ = env.step(action)
            if done:
                    break
        
        entropy = np.linalg.norm(np.asarray(final_state) - np.asarray(obs[0]))
        intrinsic_reward = fuzzer.train_rnd(sequences)    
        fuzzer.further_mutation(states, episode_reward, entropy, intrinsic_reward, final_state, states)  
        i += 1
        pbar.update(1)

    fuzzer.count = [5] * len(fuzzer.corpus)
    fuzzer.original = copy.deepcopy(fuzzer.corpus)

    start_fuzz_time = time.time()
    current_time = time.time()
    seedcount = 0
    fuzz_selection_log = []
    
    print(f"\n[Goal] Collecting data. Save Physics: {args.save_physics}")
    
    # --- Fuzzing Loop ---
    while current_time - start_fuzz_time < (3600 * 12) and len(fuzzer.corpus) > 0 and seedcount<300:
        seedcount += 1
        selected_info = fuzzer.get_pose()
        states = selected_info['seed_state']
        current_mutation_depth = selected_info['depth']

        mutate_states = fuzzer.mutation(states)
        state = None
        episode_reward = 0.0
        
        # 重置环境
        # 注意：env.reset(mutate_states) 返回的是 observation
        # 此时环境 seed 已经被环境内部逻辑或包装器处理
        # 为了物理轨迹的 seed 对齐，我们需要记录当前的 mutate_states 也就是 seed
        obs = env.reset(mutate_states)
        
        rnd_sequences = [obs[0]] 
        todynet_sequences = []   
        current_ep_transitions = []
        
        # [新增] 当前 Episode 的物理轨迹缓存
        current_episode_physics = []
        
        total_x_pos_sum = 0.0
        total_abs_angle_sum = 0.0
        episode_steps = 0
        
        # 初始帧物理状态 (Step 0)
        if args.save_physics:
            real_env = get_real_unwrapped_env(env)
            if real_env:
                init_phys = extract_physics_state(real_env)
                if init_phys:
                    current_episode_physics.append(init_phys)

        for _ in range(args.n_timesteps):
            current_obs = obs[0].copy()
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            current_action = action[0]
            
            vec_28d = np.concatenate((current_obs, current_action))
            todynet_sequences.append(vec_28d)

            obs, reward, done, _ = env.step(action)
            
            # [新增] 在 step 之后立即捕获物理状态
            if args.save_physics:
                real_env = get_real_unwrapped_env(env)
                if real_env:
                    phys_snapshot = extract_physics_state(real_env)
                    if phys_snapshot:
                        current_episode_physics.append(phys_snapshot)
            
            if args.save_transitions:
                current_ep_transitions.append((current_obs, current_action, reward[0], obs[0].copy(), done[0]))
            
            # 这里也用了 get_real_unwrapped_env 用于计算 bd_dist
            # 如果不开启 save_physics，这里会重复调用一次，无伤大雅
            real_env_stats = get_real_unwrapped_env(env)
            raw_x_pos = real_env_stats.hull.position[0]
            raw_angle = real_env_stats.hull.angle
            total_x_pos_sum += raw_x_pos
            total_abs_angle_sum += abs(raw_angle)
            episode_steps += 1
            
            rnd_sequences.append(obs[0])
            episode_reward += reward[0]
            if done:
                break
        
        bd_dist = total_x_pos_sum / max(1, episode_steps)
        bd_mean_angle = total_abs_angle_sum / max(1, episode_steps)
        
        intrinsic_reward = fuzzer.train_rnd(rnd_sequences)
        entropy = np.linalg.norm(np.asarray(obs[0]) - np.asarray(fuzzer.final_state))
        
        did_crash = False
        if done or episode_reward < 10:
            fuzzer.add_crash(mutate_states)
            did_crash = True
        else:
            condition = False
            if args.guide:
                condition = intrinsic_reward > intrins_theta or episode_reward < fuzzer.current_reward or entropy > entropy_theta
            else:
                condition = episode_reward < fuzzer.current_reward or entropy > entropy_theta
            if condition:
                fuzzer.further_mutation(copy.deepcopy(mutate_states), episode_reward, entropy, intrinsic_reward, final_state, fuzzer.current_original)
        
        # [新增] 如果 Crash 且开启了物理保存，则存储该轨迹
        if did_crash and args.save_physics:
            # 只有当轨迹长度足够时才保存（可选过滤）
            if len(current_episode_physics) > 20:
                all_crash_physics_trajectories.append({
                    "seed": mutate_states,  # 在这里 mutate_states 充当了 seed 的角色
                    "trajectory": current_episode_physics
                })
        
        if args.save_transitions:
            if did_crash:
                if len(crash_transitions) < TARGET_CRASH_COUNT:
                    crash_transitions.extend(current_ep_transitions)
            else:
                if len(success_transitions) < TARGET_SUCCESS_COUNT:
                    success_transitions.extend(current_ep_transitions)
            
            if seedcount % 100 == 0:
                c_len = len(crash_transitions)
                s_len = len(success_transitions)
                p_len = len(all_crash_physics_trajectories)
                print(f"Seeds: {seedcount} | Fail Steps: {c_len} | Physics Trajs: {p_len}")

        # === TodyNet 数据收集 (带软上限) ===
        if args.save_data:
            label = 1 if did_crash else 0
            
            # 判断是否收集
            collect_this = False
            if label == 1:
                # Crash: 总是收集
                collect_this = True
            else:
                # Success: 仅当未达到软上限时收集
                if todynet_success_count < TODYNET_SUCCESS_SOFT_CAP:
                    collect_this = True
            
            if collect_this:
                wins, labels = process_episode_data(todynet_sequences, label, args.window_size)
                if wins is not None and len(wins) > 0:
                    all_window_data.append(wins)
                    all_label_data.append(labels)
                    if label == 0:
                        todynet_success_count += 1
        
        fuzz_selection_log.append({
            'seed_state': selected_info['seed_state'],
            'mutate_state': mutate_states,
            'parent_depth': current_mutation_depth,
            'did_crash': did_crash,
            'elapsed_time': time.time() - start_fuzz_time,
            'bd_distance': bd_dist,      
            'bd_mean_angle': bd_mean_angle 
        })
        
        current_time = time.time()

    # === 保存 Transitions ===
    if args.save_transitions:
        final_transitions = crash_transitions + success_transitions
        random.shuffle(final_transitions)
        
        trans_file = os.path.join(result_path, 'transitions.pkl')
        print(f"Saving {len(final_transitions)} total transitions to {trans_file}...")
        with open(trans_file, 'wb') as f_t:
            pickle.dump(final_transitions, f_t, protocol=pickle.HIGHEST_PROTOCOL)
        print("Transitions saved.")
        
    # === [新增] 保存物理状态轨迹 ===
    if args.save_physics:
        phys_file = os.path.join(result_path, 'physics_trajectory.pkl')
        print(f"Saving {len(all_crash_physics_trajectories)} physics trajectories to {phys_file}...")
        with open(phys_file, 'wb') as f_p:
            pickle.dump(all_crash_physics_trajectories, f_p, protocol=pickle.HIGHEST_PROTOCOL)
        print("Physics trajectories saved.")

    # === 保存 TodyNet 数据 (应用目标3000条, 30% Crash) ===
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

    crash_file = 'cure_crash.pkl' if args.guide else 'ablated_crash.pkl'
    with open(os.path.join(result_path, crash_file), 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    log_file_name = os.path.join(result_path, 'selection_log.pkl')
    with open(log_file_name, 'wb') as handle:
        pickle.dump(fuzz_selection_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not args.no_render:
        if args.n_envs == 1 and "Bullet" not in env_id and not is_atari and isinstance(env, VecEnv):
            while isinstance(env, VecEnvWrapper):
                env = env.venv
            if isinstance(env, DummyVecEnv):
                env.envs[0].env.close()
            else:
                env.close()
        else:
            env.close()

if __name__ == "__main__":
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- start time: {start_time_str} ---")
    main()
    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- finish time: {end_time_str} ---")
    duration = end_time - start_time
    print(f"--- total time: {duration} ---")