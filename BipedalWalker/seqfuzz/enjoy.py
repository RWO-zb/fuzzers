import argparse, importlib, os, sys, time, copy, tqdm, pickle, yaml
import numpy as np
import torch as th
import torch # 确保导入 torch
import random # 确保导入 random
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecVideoRecorder
import utils.import_envs
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from fuzz.fuzz import fuzzing
from datetime import datetime
import joblib
from tapnet import predict_siamese, Hyperparameter

# ==========================================
# [新增] TodyNet 数据收集辅助函数 (来自 mdpfuzz)
# ==========================================
def process_episode_data(sequence, label, window_size):
    """
    TodyNet 严格采样: Success随机取1个，Failure取最后1个
    """
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

def balance_and_save_data(X_list, y_list, output_dir, dataset_name, window_size, target_total=3000, target_crash_ratio=0.30):
    """
    平衡数据：强制 Crash 占比 target_crash_ratio (0.30)，且总数限制为 target_total (3000)
    """
    if not X_list:
        return
    
    print(f"\n[TodyNet Data] Processing balancing to {target_total} samples (Target Crash Ratio: {target_crash_ratio:.0%})...")
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    indices_fail = np.where(y_all == 1)[0]
    indices_succ = np.where(y_all == 0)[0]
    
    # 目标计算
    n_crash_target = int(target_total * target_crash_ratio)
    n_succ_target = target_total - n_crash_target
    
    print(f"  Raw Collected: Fail={len(indices_fail)}, Success={len(indices_succ)}")
    print(f"  Target: Fail={n_crash_target}, Success={n_succ_target}")

    # 1. 采样 Crash
    if len(indices_fail) >= n_crash_target:
        final_fail = np.random.choice(indices_fail, size=n_crash_target, replace=False)
    else:
        print(f"  [Warning] Not enough crash samples! Keeping all {len(indices_fail)}.")
        final_fail = indices_fail
        
    # 2. 采样 Success
    if len(indices_succ) >= n_succ_target:
        final_succ = np.random.choice(indices_succ, size=n_succ_target, replace=False)
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

# --- 获取底层环境以访问物理状态 ---
def get_real_unwrapped_env(env):
    current_env = env
    while hasattr(current_env, 'venv'):
        current_env = current_env.venv
    while hasattr(current_env, 'env'):
        current_env = current_env.env
    if hasattr(current_env, 'envs'):
        return current_env.envs[0].unwrapped
    if hasattr(current_env, 'unwrapped'):
        return current_env.unwrapped
    return current_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="../rl-trained-agents/")
    parser.add_argument("--algo", help="RL Algorithm", default="tqc", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n_timesteps", help="number of timesteps", default=300, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=True, help="Do not render the environment (useful for tests)"
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
    parser.add_argument("--em", action="store_true", default=True)
    
    # [新增] 参数：是否收集 TodyNet 数据
    parser.add_argument("--save-data", action="store_true", default=True, help="Save TodyNet training data")
    parser.add_argument("--window-size", type=int, default=25, help="Sliding window size")

    args = parser.parse_args()
    
    # --- 创建结果文件夹 ---
    now_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    result_folder = f"{now_str}_seed_{args.seed}"
    result_path = os.path.join('results', result_folder)
    os.makedirs(result_path, exist_ok=True)
    log_file_path = os.path.join(result_path, 'fuzz.txt')
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
        print(f"Loading latest experiment, id={args.exp_id}")

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
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
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

    np.random.seed(2021)
    states = np.random.randint(low=1, high=4, size=15)
    obs = env.reset(states)

    siamese_model = predict_siamese.load_tapnet_mode()
    siamese_model.cuda()
    siamese_model.load_state_dict(torch.load(r'./tapnet/data/weights/tapnet.pkl'))
    siamese_model.eval()
    bench_noCrash = Hyperparameter.bench_noCrash
    if len(bench_noCrash) == 0:
        bench_noCrash = torch.zeros((1, Hyperparameter.Step, Hyperparameter.Dimension)).cuda()
    else:
         bench_noCrash = torch.FloatTensor(np.array(bench_noCrash)).cuda()
         if len(bench_noCrash.shape) == 2:
            bench_noCrash = bench_noCrash.unsqueeze(0)
   
    print('nodel:')
    print(siamese_model)

    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    episode_rewards, episode_lengths = [], []
    ep_len = 0
    successes = []
    fuzzer = fuzzing()
    seeds_num = 200000
    i = 0
    pbar = tqdm.tqdm(total=seeds_num)
    
    # --- [新增] TodyNet 数据收集容器 ---
    all_window_data = [] 
    all_label_data = []
    todynet_success_count = 0
    # -------------------------------

    # --- Corpus Generation Loop ---
    while i < seeds_num:
        states = np.random.randint(low=1, high=4, size=15)
        state = None
        episode_reward = 0.0
        obs = env.reset(states)
        sequences = [obs[0]]
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            sequences.append(obs[0])
            episode_reward += reward[0]
            if done:
                break
        if not done:
            state = None
            episode_reward_mutate = 0.0
            delta_states = np.random.choice(2, 15, p=[0.9, 0.1])
            if np.sum(delta_states) == 0:
                delta_states[0] = 1
            mutate_states = states + delta_states
            mutate_states = np.remainder(mutate_states, 4)
            mutate_states = np.clip(mutate_states, 1, 3)

            obs = env.reset(mutate_states)
            print('mutate states ', mutate_states)

            for _ in range(args.n_timesteps):
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                episode_reward_mutate += reward[0]
                if done:
                    break
            entropy = np.abs(episode_reward_mutate - episode_reward) / np.sum(delta_states)
            cvg = fuzzer.state_coverage(sequences)
            fuzzer.further_mutation(states, episode_reward, entropy, cvg, states,0)
            print(entropy, episode_reward, episode_reward_mutate, done, cvg)
            i += 1
            pbar.update(1)

    
    with open(os.path.join(result_path, 'corpus_EM.pkl'), 'wb') as handle:
        pickle.dump(fuzzer.corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(result_path, 'rewards_EM.pkl'), 'wb') as handle:
        pickle.dump(fuzzer.rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(result_path, 'entropy_EM.pkl'), 'wb') as handle:
        pickle.dump(fuzzer.entropy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(result_path, 'cvg_EM.pkl'), 'wb') as handle:
        pickle.dump(fuzzer.coverage, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    fuzzer.count = [5] * len(fuzzer.corpus)
    fuzzer.original = copy.deepcopy(fuzzer.corpus)
    mutation_log = [] 

    # HACK: start fuzzing
    start_fuzz_time = time.time()
    cvg_threshold = 0.02

    current_time = time.time()
    pbar1 = tqdm.tqdm(total=seeds_num)
    time_of_env = 0
    time_of_fuzzer = 0
    time_of_DynEM = 0

    successObs = open(os.path.join(result_path, 'noCrashStateSeqV2.txt'), mode='a')
    failObs = open(os.path.join(result_path, 'crashStateSeqV2.txt'), mode='a')
    crashF_40 = open(os.path.join(result_path, 'crashStateSeqV2_40.txt'), mode='a')
    noCrashF_40 = open(os.path.join(result_path, 'noCrashStateSeqV2_40.txt'), mode='a')
    timeStamp = open(os.path.join(result_path, 'timeStamp.txt'), mode='a')
    seedcount = 0
    
    # --- Fuzzing Loop ---
    while current_time - start_fuzz_time < 3600 * 100 and len(fuzzer.corpus) > 0 :
        is_crash = False
        seedcount+=1
        output_obs = []
        temp1_time = time.time()
        states = fuzzer.get_pose()
        mutate_states = fuzzer.mutation(states)
        current_gen = fuzzer.current_generation + 1 
        state = None
        episode_reward = 0.0
        obs = env.reset(mutate_states)
        sequences = [obs[0]]
        
        # --- [新增] 初始化行为特征和 TodyNet 序列容器 ---
        total_x_pos_sum = 0.0
        total_abs_angle_sum = 0.0
        episode_steps = 0
        current_episode_transitions_for_todynet = [] # 存储 (s, a) 对
        # ---------------------------------------------

        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            
            # --- [新增] 记录 TodyNet 所需的 (State, Action) ---
            # obs 是当前的 State，action 是将要执行的动作
            # 兼容处理：确保 obs 和 action 格式正确
            curr_obs_copy = obs.copy() if isinstance(obs, np.ndarray) else obs
            curr_action_copy = action.copy() if isinstance(action, np.ndarray) else action
            current_episode_transitions_for_todynet.append((curr_obs_copy, curr_action_copy))
            # -----------------------------------------------

            obs, reward, done, infos = env.step(action)
            sequences.append(obs[0])
            
            # --- [新增] 收集行为特征 (Distance & Angle) ---
            real_env = get_real_unwrapped_env(env)
            if real_env is not None and hasattr(real_env, 'hull'):
                raw_x_pos = real_env.hull.position[0]
                raw_angle = real_env.hull.angle
                total_x_pos_sum += raw_x_pos
                total_abs_angle_sum += abs(raw_angle)
            episode_steps += 1
            # ---------------------------------------------
            
            if not args.no_render:
                env.render("human")
            episode_reward += reward[0]

            output_obs.append(obs[0])
            if (len(output_obs) == Hyperparameter.Step): 
                ret = predict_siamese.predict_once(siamese_model, bench_noCrash, output_obs)
                if ret == 1:
                    print('end')
                else:
                    print('continue')
            
            # [Fix] 如果 done 了，提前 break，不需要等 for 循环结束
            if done:
                break

        # --- [新增] 计算本回合的 BD 指标 ---
        bd_dist = total_x_pos_sum / max(1, episode_steps)
        bd_mean_angle = total_abs_angle_sum / max(1, episode_steps)
        
        temp2_time = time.time()
        time_of_env += temp2_time - temp1_time
        cvg = fuzzer.state_coverage(sequences)
        temp3_time = time.time()
        time_of_DynEM += temp3_time - temp2_time
        local_sensitivity = np.abs(episode_reward - fuzzer.current_reward)
        if done or episode_reward < 10:
            is_crash = True
            if len(output_obs) == Hyperparameter.Step:
                for i in range(len(output_obs)):
                    outputStr = ''
                    for d in output_obs[i]:
                        outputStr = outputStr + str(d) + ', '
                    crashF_40.write(outputStr)
                    crashF_40.write('\n')
                crashF_40.write('######')
                crashF_40.write('\n')

                current_time = time.time()
                s = 'fail_40: '
                timeStamp.write(s)
                timeStamp.write(str(current_time))
                timeStamp.write('\n')

            else:
                for i in range(len(output_obs)):
                    outputStr = ''
                    for d in output_obs[i]:
                        outputStr = outputStr + str(d) + ', '
                    failObs.write(outputStr)
                    failObs.write('\n')
                failObs.write('######')
                failObs.write('\n')

                current_time = time.time()
                s = 'fail: '
                timeStamp.write(s)
                timeStamp.write(str(current_time))
                timeStamp.write('\n')


            pbar1.update(1)
            fuzzer.add_crash(mutate_states)
            print('Found: ', len(fuzzer.result))
        elif args.em:
            if len(output_obs) == Hyperparameter.Step:
                for i in range(len(output_obs)):
                    outputStr = ''
                    for d in output_obs[i]:
                        outputStr = outputStr + str(d) + ', '
                    noCrashF_40.write(outputStr)
                    noCrashF_40.write('\n')
                noCrashF_40.write('######')
                noCrashF_40.write('\n')
                current_time = time.time()

                s = 'success_40: '
                timeStamp.write(s)
                timeStamp.write(str(current_time))
                timeStamp.write('\n')
            else:
                for i in range(len(output_obs)):
                    outputStr = ''
                    for d in output_obs[i]:
                        outputStr = outputStr + str(d) + ', '
                    successObs.write(outputStr)
                    successObs.write('\n')
                successObs.write('######')
                successObs.write('\n')
                current_time = time.time()

                s = 'success: '
                timeStamp.write(s)
                timeStamp.write(str(current_time))
                timeStamp.write('\n')

            if cvg < cvg_threshold or episode_reward < fuzzer.current_reward:
                current_pose = copy.deepcopy(mutate_states)
                orig_pose = fuzzer.current_original
                fuzzer.further_mutation(current_pose, episode_reward, local_sensitivity, cvg, orig_pose,current_gen)
        else:
            if len(output_obs) == Hyperparameter.Step:
                for i in range(len(output_obs)):
                    outputStr = ''
                    for d in output_obs[i]:
                        outputStr = outputStr + str(d) + ', '
                    noCrashF_40.write(outputStr)
                    noCrashF_40.write('\n')
                noCrashF_40.write('######')
                noCrashF_40.write('\n')
                current_time = time.time()

                s = 'success_40: '
                timeStamp.write(s)
                timeStamp.write(str(current_time))
                timeStamp.write('\n')
            else:
                for i in range(len(output_obs)):
                    outputStr = ''
                    for d in output_obs[i]:
                        outputStr = outputStr + str(d) + ', '
                    successObs.write(outputStr)
                    successObs.write('\n')
                successObs.write('######')
                successObs.write('\n')
                current_time = time.time()

                s = 'success: '
                timeStamp.write(s)
                timeStamp.write(str(current_time))
                timeStamp.write('\n')

            if episode_reward < fuzzer.current_reward:
                current_pose = copy.deepcopy(mutate_states)
                orig_pose = fuzzer.current_original
                fuzzer.further_mutation(current_pose, episode_reward, local_sensitivity, cvg, orig_pose,current_gen)
        
        # --- [新增] TodyNet 数据收集逻辑 ---
        if args.save_data:
            TODYNET_SUCCESS_CAP = 3000
            collect_this = False
            if is_crash:
                collect_this = True
            else:
                if todynet_success_count < TODYNET_SUCCESS_CAP:
                    collect_this = True
            
            if collect_this:
                # 构造序列: vec = [state, action]
                todynet_seq = []
                for (s, a) in current_episode_transitions_for_todynet:
                    # s 可能为 (1, 24) 或 (24,), a 可能为 (1, 4) 或 (4,)
                    # 确保展平
                    s_flat = s.flatten()
                    a_flat = a.flatten()
                    vec = np.concatenate([s_flat, a_flat])
                    todynet_seq.append(vec)
                
                label = 1 if is_crash else 0
                wins, labels = process_episode_data(todynet_seq, label, args.window_size)
                if wins is not None and len(wins) > 0:
                    all_window_data.append(wins)
                    all_label_data.append(labels)
                    if not is_crash:
                        todynet_success_count += 1
        # -----------------------------------

        # --- [新增] 记录日志包含行为特征 ---
        log_entry = {
            'state': copy.deepcopy(mutate_states), 
            'generation': current_gen,             
            'crashed': is_crash,
            'timestamp': time.time() - start_fuzz_time,
            'bd_distance': bd_dist,      
            'bd_mean_angle': bd_mean_angle 
        }
        mutation_log.append(log_entry)
        # --- 日志记录结束 ---
        
        current_time = time.time()
        time_of_fuzzer += current_time - temp2_time
        print('total reward: ', episode_reward, ', coverage: ', cvg, ', passed time: ', current_time - start_fuzz_time, ', corpus size: ', len(fuzzer.corpus), 'time_of_fuzzer: ', time_of_fuzzer, 'time_of_env: ', time_of_env)
        
        # 打印当前 TodyNet 收集进度
        if seedcount % 10 == 0:
            print(f"[TodyNet Info] Collected: {todynet_success_count} success samples, Total collected episodes: {len(all_window_data)}")
    
    if args.em:
        file_name = os.path.join(result_path, 'crash_EM.pkl')
    else:
        file_name = os.path.join(result_path, 'crash_noEM.pkl')
    with open(file_name, 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # --- 保存完整的变异日志 ---
    with open(os.path.join(result_path, 'all_run_seeds_0.pkl'), 'wb') as handle:
        pickle.dump(mutation_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # --- [新增] 保存 TodyNet 数据 ---
    if args.save_data:
        balance_and_save_data(
            all_window_data, 
            all_label_data, 
            result_path, 
            "BipedalWalkerHC", 
            args.window_size, 
            target_total=3000, 
            target_crash_ratio=0.30
        )
    # -------------------------------


    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

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
    print(f"--- end time: {end_time_str} ---")
    duration = end_time - start_time
    print(f"--- total time: {duration} ---")