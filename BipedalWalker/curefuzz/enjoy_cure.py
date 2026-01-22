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
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecNormalize
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from fuzz.cure_fuzz import CureFuzz

# [Helper] 提取物理状态 (用于 Box2D 物理重放)
def extract_physics_state(real_env):
    if not hasattr(real_env, 'hull') or not hasattr(real_env, 'legs'):
        return None
    hull = real_env.hull
    legs = real_env.legs 
    state_dict = {
        "hull_pos": (hull.position[0], hull.position[1]),
        "hull_angle": hull.angle,
        "hull_lin_vel": (hull.linearVelocity[0], hull.linearVelocity[1]),
        "hull_ang_vel": hull.angularVelocity,
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

# [Helper] TodyNet 处理
def process_episode_data(sequence, label, window_size):
    seq_len = len(sequence)
    if seq_len < window_size:
        return None, None
    seq_array = np.array(sequence) 
    windows = []
    labels = []
    if label == 0:
        max_idx = seq_len - window_size
        rand_idx = random.randint(0, max_idx)
        win = seq_array[rand_idx : rand_idx + window_size]
        win = win.transpose() 
        windows.append(win)
        labels.append(0)
    else:
        win = seq_array[-window_size:] 
        win = win.transpose()
        windows.append(win)
        labels.append(1)
    return np.array(windows), np.array(labels)

# [Helper] TodyNet 平衡与保存
def balance_and_save_data(X_list, y_list, output_dir, dataset_name, window_size, target_total=3000, target_crash_ratio=0.30):
    if not X_list: return
    print(f"\n[TodyNet Data] Balancing to {target_total} samples...")
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    indices_fail = np.where(y_all == 1)[0]
    indices_succ = np.where(y_all == 0)[0]
    target_n_fail = int(target_total * target_crash_ratio)
    target_n_succ = target_total - target_n_fail
    
    final_fail = np.random.choice(indices_fail, size=target_n_fail, replace=False) if len(indices_fail) >= target_n_fail else indices_fail
    final_succ = np.random.choice(indices_succ, size=target_n_succ, replace=False) if len(indices_succ) >= target_n_succ else indices_succ
    
    final_indices = np.concatenate([final_fail, final_succ])
    np.random.shuffle(final_indices)
    X_balanced, y_balanced = X_all[final_indices], y_all[final_indices]

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
    print(f"[TodyNet Data] Saved {total} samples to {save_path}")

# [Helper] 获取底层环境 (用于物理属性)
def get_real_unwrapped_env(env):
    current_env = env
    while hasattr(current_env, 'venv'):
        current_env = current_env.venv
    if hasattr(current_env, 'envs'):
        return current_env.envs[0].unwrapped
    return current_env.unwrapped

# [CRITICAL HELPER] 获取反归一化的原始 Observation
def get_raw_obs_from_env(env, obs):
    """
    如果环境使用了 VecNormalize，将 obs 反归一化为原始物理数值。
    """
    # 1. 找到 VecNormalize 对象
    norm_env = env
    # 处理嵌套情况 (DummyVecEnv -> VecNormalize)
    if hasattr(norm_env, 'venv') and isinstance(norm_env.venv, VecNormalize):
        norm_env = norm_env.venv
    # 处理直接是 VecNormalize 的情况
    elif isinstance(norm_env, VecNormalize):
        pass
    else:
        # 没有归一化，直接返回
        return obs

    # 2. 调用 unnormalize_obs
    return norm_env.unnormalize_obs(obs)

def main():
    parser = argparse.ArgumentParser()
    # ... (保持原有的参数定义不变)
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
    parser.add_argument("--seed_number", help="Number of seeds", default=1000, type=int)
    parser.add_argument("--save-data", action="store_true", default=False, help="Save TodyNet training data")
    parser.add_argument("--save-transitions", action="store_true", default=False, help="Save RL transitions")
    parser.add_argument("--save-physics", action="store_true", default=False, help="Save full physics state trajectories")
    parser.add_argument("--window-size", type=int, default=25, help="Sliding window size")
    parser.add_argument("--dataset-name", type=str, default="BipedalWalkerHC", help="Dataset name prefix")
    args = parser.parse_args()
    
    # ... (日志和路径设置代码保持不变) ...
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

    # ... (环境加载代码保持不变) ...
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
    
    all_window_data = [] 
    all_label_data = []
    todynet_success_count = 0 
    crash_transitions = []
    success_transitions = []
    all_crash_physics_trajectories = []
    
    TARGET_CRASH_COUNT = 10000
    TARGET_SUCCESS_COUNT = 90000
    TODYNET_SUCCESS_SOFT_CAP = 5000

    pbar = tqdm.tqdm(total=seeds_num)
    start_corpus_time = time.time()
    i = 0
    
    # --- Corpus Generation ---
    # (这部分通常作为热身，不需要保存 transition，略过修改)
    while i < seeds_num and (time.time() - start_corpus_time) <= (3600*2):
        # ... (保持原有的 Corpus 生成代码不变) ...
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
            if done: break
        final_state = sequences[-2]
        delta_states = np.random.choice(2, 15, p=[0.9, 0.1])
        if np.sum(delta_states) == 0: delta_states[0] = 1
        mutate_states = np.clip(np.remainder(states + delta_states, 4), 1, 3)
        obs = env.reset(mutate_states)
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, _, done, _ = env.step(action)
            if done: break
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
    
    print(f"\n[Goal] Collecting RAW Transitions. Save Physics: {args.save_physics}")
    
    # --- Fuzzing Loop ---
    while current_time - start_fuzz_time < (3600 * 12) and len(fuzzer.corpus) > 0:
        seedcount += 1
        selected_info = fuzzer.get_pose()
        states = selected_info['seed_state']
        current_mutation_depth = selected_info['depth']

        mutate_states = fuzzer.mutation(states)
        state = None
        episode_reward = 0.0
        
        obs = env.reset(mutate_states)
        
        rnd_sequences = [obs[0]] 
        todynet_sequences = []   
        current_ep_transitions = []
        current_episode_physics = []
        
        total_x_pos_sum = 0.0
        total_abs_angle_sum = 0.0
        episode_steps = 0
        
        if args.save_physics:
            real_env = get_real_unwrapped_env(env)
            if real_env:
                init_phys = extract_physics_state(real_env)
                if init_phys: current_episode_physics.append(init_phys)

        for _ in range(args.n_timesteps):
            # [Fix 1] 获取当前步的 RAW Observation
            # obs 已经是 Normalized 的了，我们需要反推或者直接用 get_raw (如果 unnormalize 可用)
            # 注意：SB3 VecNormalize 通常在 step 返回时就归一化了。我们需要用 get_raw_obs_from_env 还原。
            raw_current_obs = get_raw_obs_from_env(env, obs[0].copy())
            
            # 预测动作 (使用 Normalized obs)
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            current_action = action[0]
            
            # TodyNet 序列使用 Normalized 数据 (这没问题，TodyNet通常基于模型视角)
            vec_28d = np.concatenate((obs[0], current_action))
            todynet_sequences.append(vec_28d)

            # 执行环境交互
            # next_obs 是 Normalized 的
            next_obs, reward, done, _ = env.step(action)
            
            # [Fix 2] 获取下一步的 RAW Observation
            raw_next_obs = get_raw_obs_from_env(env, next_obs[0].copy())
            
            if args.save_physics:
                real_env = get_real_unwrapped_env(env)
                if real_env:
                    phys_snapshot = extract_physics_state(real_env)
                    if phys_snapshot: current_episode_physics.append(phys_snapshot)
            
            if args.save_transitions:
                # [Fix 3] 保存 RAW data!
                # 格式: (Raw Obs, Action, Reward, Raw Next Obs, Done)
                current_ep_transitions.append((raw_current_obs, current_action, reward[0], raw_next_obs, done[0]))
            
            real_env_stats = get_real_unwrapped_env(env)
            raw_x_pos = real_env_stats.hull.position[0]
            raw_angle = real_env_stats.hull.angle
            total_x_pos_sum += raw_x_pos
            total_abs_angle_sum += abs(raw_angle)
            episode_steps += 1
            
            rnd_sequences.append(next_obs[0]) # RND 使用 Normalized obs
            episode_reward += reward[0]
            
            # 更新 obs
            obs = next_obs
            
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
        
        if did_crash and args.save_physics:
            if len(current_episode_physics) > 20:
                all_crash_physics_trajectories.append({
                    "seed": mutate_states,  
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
                print(f"Seeds: {seedcount} | Crash Samples: {c_len} | Physics Trajs: {len(all_crash_physics_trajectories)}")

        # TodyNet 保存逻辑 (保持不变)
        if args.save_data:
            label = 1 if did_crash else 0
            collect_this = False
            if label == 1: collect_this = True
            else:
                if todynet_success_count < TODYNET_SUCCESS_SOFT_CAP: collect_this = True
            
            if collect_this:
                wins, labels = process_episode_data(todynet_sequences, label, args.window_size)
                if wins is not None and len(wins) > 0:
                    all_window_data.append(wins)
                    all_label_data.append(labels)
                    if label == 0: todynet_success_count += 1
        
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

    # --- 保存结果 ---
    if args.save_transitions:
        save_data = {
            "crash": crash_transitions,
            "success": success_transitions,
            "is_raw": True # 标记这是 Raw Data
        }
        trans_file = os.path.join(result_path, 'transitions.pkl')
        print(f"Saving labeled RAW transitions to {trans_file}...")
        with open(trans_file, 'wb') as f_t:
            pickle.dump(save_data, f_t, protocol=pickle.HIGHEST_PROTOCOL)
        print("Transitions saved.")
        
    if args.save_physics:
        phys_file = os.path.join(result_path, 'physics_trajectory.pkl')
        with open(phys_file, 'wb') as f_p:
            pickle.dump(all_crash_physics_trajectories, f_p, protocol=pickle.HIGHEST_PROTOCOL)
        print("Physics trajectories saved.")

    if args.save_data:
        balance_and_save_data(all_window_data, all_label_data, result_path, args.dataset_name, args.window_size)

    crash_file = 'cure_crash.pkl' if args.guide else 'ablated_crash.pkl'
    with open(os.path.join(result_path, crash_file), 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    log_file_name = os.path.join(result_path, 'selection_log.pkl')
    with open(log_file_name, 'wb') as handle:
        pickle.dump(fuzz_selection_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not args.no_render:
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