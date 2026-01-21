import argparse, importlib, os, sys, time, copy, tqdm, pickle, yaml
import numpy as np
import torch as th
import torch # 确保导入 torch
import random # 确保导入 random
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecVideoRecorder, VecNormalize
import utils.import_envs
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from fuzz.fuzz import fuzzing
from datetime import datetime
import joblib
from tapnet import predict_siamese, Hyperparameter

# ==========================================
# [新增] 辅助函数：获取 Raw Observation
# ==========================================
def get_raw_obs(env, obs):
    """
    如果环境被 VecNormalize 包装，则进行反归一化以获取原始物理数值。
    """
    norm_env = env
    # 处理嵌套情况 (DummyVecEnv -> VecNormalize)
    if hasattr(norm_env, 'venv'):
        if hasattr(norm_env.venv, 'envs') and len(norm_env.venv.envs) > 0:
             possible_norm = norm_env.venv
             if isinstance(possible_norm, VecNormalize):
                 return possible_norm.unnormalize_obs(obs)
    
    # 直接检查
    if isinstance(norm_env, VecNormalize):
        return norm_env.unnormalize_obs(obs)
    
    # 尝试从 unwrapped env 获取
    return obs

# ==========================================
# [新增] TodyNet 数据处理函数 (来自 mdpfuzz)
# ==========================================
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

def balance_and_save_data(X_list, y_list, output_dir, dataset_name, window_size, target_total=3000, target_crash_ratio=0.30):
    if not X_list:
        return
    
    print(f"\n[TodyNet Data] Processing balancing to {target_total} samples (Target Crash Ratio: {target_crash_ratio:.0%})...")
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    indices_fail = np.where(y_all == 1)[0]
    indices_succ = np.where(y_all == 0)[0]
    
    n_crash_target = int(target_total * target_crash_ratio)
    n_succ_target = target_total - n_crash_target
    
    if len(indices_fail) >= n_crash_target:
        final_fail = np.random.choice(indices_fail, size=n_crash_target, replace=False)
    else:
        final_fail = indices_fail
        
    if len(indices_succ) >= n_succ_target:
        final_succ = np.random.choice(indices_succ, size=n_succ_target, replace=False)
    else:
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
    
    print(f"[TodyNet Data] Saved {total} samples to {save_path}")

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
    
    # [新增] 数据收集参数
    parser.add_argument("--save-data", action="store_true", default=True, help="Save TodyNet & Transition data")
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
    siamese_model.load_state_dict(th.load(r'./tapnet/data/weights/tapnet.pkl'))
    siamese_model.eval()
    bench_noCrash = Hyperparameter.bench_noCrash
    if len(bench_noCrash) == 0:
        bench_noCrash = th.zeros((1, Hyperparameter.Step, Hyperparameter.Dimension)).cuda()
    else:
         bench_noCrash = th.FloatTensor(np.array(bench_noCrash)).cuda()
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
    seeds_num = 1000
    i = 0
    pbar = tqdm.tqdm(total=seeds_num)

    # [新增] 数据收集容器
    all_window_data = [] 
    all_label_data = []
    todynet_success_count = 0
    
    # [新增] Transition Lists (用于 Retrain)
    crash_transitions = []
    success_transitions = []
    TARGET_CRASH_TRANS = 10000
    TARGET_SUCCESS_TRANS = 90000

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
    
    while current_time - start_fuzz_time < 3600 * 12 and len(fuzzer.corpus) > 0 :
        is_crash = False
        seedcount+=1
        output_obs = []
        temp1_time = time.time()
        states = fuzzer.get_pose()
        mutate_states = fuzzer.mutation(states)
        current_gen = fuzzer.current_generation + 1 
        state = None
        episode_reward = 0.0
        
        # [Fix] Reset and get Raw
        obs = env.reset(mutate_states)
        sequences = [obs[0]]
        
        # [新增] 本 Episode 的临时数据容器
        current_ep_transitions_raw = [] 
        
        for _ in range(args.n_timesteps):
            # [新增] 获取 Step 前的 Raw Obs
            curr_obs_raw = get_raw_obs(env, obs[0].copy())
            
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            
            # Step
            next_obs, reward, done, infos = env.step(action)
            
            # [新增] 获取 Step 后的 Raw Obs
            next_obs_raw = get_raw_obs(env, next_obs[0].copy())
            
            # [新增] 收集 Raw Transition: (s, a, r, s', d)
            # action[0] 取出 scalar/array 动作
            current_ep_transitions_raw.append((curr_obs_raw, action[0], reward[0], next_obs_raw, done))

            # 原逻辑继续...
            obs = next_obs
            sequences.append(obs[0])
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
            
            if done:
                break

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
        
        # --- [新增] 数据保存逻辑 (插入到循环末尾) ---
        if args.save_data:
            # 1. 收集 Transitions
            if is_crash:
                if len(crash_transitions) < TARGET_CRASH_TRANS:
                    crash_transitions.extend(current_ep_transitions_raw)
            else:
                if len(success_transitions) < TARGET_SUCCESS_TRANS:
                    success_transitions.extend(current_ep_transitions_raw)

            # 2. 收集 TodyNet 数据 (需要拼接 Raw Obs 和 Action)
            TODYNET_SUCCESS_CAP = 3000
            collect_this = True if is_crash else (todynet_success_count < TODYNET_SUCCESS_CAP)
            
            if collect_this:
                todynet_seq = []
                for t in current_ep_transitions_raw:
                    s, a, _, _, _ = t
                    # s is Raw Obs, a is Action
                    vec = np.concatenate([s, a])
                    todynet_seq.append(vec)
                
                label = 1 if is_crash else 0
                wins, labels = process_episode_data(todynet_seq, label, args.window_size)
                if wins is not None and len(wins) > 0:
                    all_window_data.append(wins)
                    all_label_data.append(labels)
                    if not is_crash: todynet_success_count += 1
        # --------------------------------------------

        current_time = time.time()
        time_of_fuzzer += current_time - temp2_time
        print('total reward: ', episode_reward, ', coverage: ', cvg, ', passed time: ', current_time - start_fuzz_time, ', corpus size: ', len(fuzzer.corpus), 'time_of_fuzzer: ', time_of_fuzzer, 'time_of_env: ', time_of_env)
        
        if seedcount % 10 == 0 and args.save_data:
             print(f"[Info] Trans(F/S): {len(crash_transitions)}/{len(success_transitions)}, TodyNet(S): {todynet_success_count}")

    if args.em:
        file_name = os.path.join(result_path, 'crash_EM.pkl')
    else:
        file_name = os.path.join(result_path, 'crash_noEM.pkl')
    with open(file_name, 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # --- [新增] 保存 Transition 和 TodyNet 数据 ---
    if args.save_data:
        # 1. 保存 TodyNet
        balance_and_save_data(all_window_data, all_label_data, result_path, "BipedalWalkerHC", args.window_size)
        
        # 2. 保存 Transitions Dict (Critical for Retrain)
        trans_file = os.path.join(result_path, 'transitions.pkl')
        print(f"Saving {len(crash_transitions)} crash / {len(success_transitions)} success transitions to {trans_file}...")
        save_payload = {
            "crash": crash_transitions, 
            "success": success_transitions, 
            "is_raw": True
        }
        with open(trans_file, 'wb') as f_t:
            pickle.dump(save_payload, f_t, protocol=pickle.HIGHEST_PROTOCOL)
    # --------------------------------------------

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