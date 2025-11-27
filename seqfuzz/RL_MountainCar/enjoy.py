import argparse, importlib, os, sys, time, copy, tqdm, pickle, yaml
import numpy as np
import torch as th
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecVideoRecorder
import utils.import_envs
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from fuzz.fuzz import fuzzing
from datetime import datetime
import joblib
# --- 引入 read_data 以加载历史数据 ---
from tapnet import predict_siamese, Hyperparameter, read_data 
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="MountainCar-v0")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--algo", help="RL Algorithm", default="dqn", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n_timesteps", help="number of timesteps", default=200, type=int)
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
    args = parser.parse_args()
    
    now_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    result_folder = f"{now_str}_seed_{args.seed}"
    if not os.path.exists('./results'):
        os.makedirs('./results')
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
    
    states = np.array([np.random.uniform(-0.6, -0.4), 0.0])
    obs = env.reset()
    env.envs[0].unwrapped.state = states
    obs = np.array([states])

    # --- TapNet 初始化与 K-Voting 数据准备 ---
    siamese_model = predict_siamese.load_tapnet_mode()
    try:
        if torch.cuda.is_available():
            siamese_model.cuda()
        if os.path.exists('./tapnet/data/weights/tapnet.pkl'):
            siamese_model.load_state_dict(torch.load(r'./tapnet/data/weights/tapnet.pkl'))
            print("Loaded TapNet weights.")
        else: 
            print("TapNet weights not found, using random init.")
    except Exception as e:
        print(f"Warning: Could not load TapNet weights: {e}. Proceeding with random weights for testing flow.")

    siamese_model.eval()

    # [关键修改] 准备 Golden Sequences (K=10)
    # 尝试从文件中读取成功的历史轨迹
    try:
        _, success_data = read_data.get_data() # 获取历史成功数据
        K = 10 # K-Voting 的 K 值
        if len(success_data) > 0:
            if len(success_data) >= K:
                # 随机采样 K 个作为 Golden Sequences
                indices = np.random.choice(len(success_data), K, replace=False)
                bench_noCrash = [success_data[i] for i in indices]
            else:
                # 如果数据不足，重复使用
                print(f"Warning: Not enough golden sequences found ({len(success_data)} < {K}). Using available ones.")
                bench_noCrash = success_data
                # 补齐到 K 个 (可选，这里暂不补齐，predict_voting 会自动处理 batch size)
        else:
            # 如果没有文件数据，使用全0初始化 (Fallback)
            print("Warning: No golden sequences found in file. Using zero tensors.")
            bench_noCrash = [np.zeros((Hyperparameter.Step, Hyperparameter.Dimension)).tolist()] * K
    except Exception as e:
        print(f"Error loading golden sequences: {e}. Using zero tensors.")
        bench_noCrash = [np.zeros((Hyperparameter.Step, Hyperparameter.Dimension)).tolist()] * 10

    # 将 bench_noCrash 转换为 Tensor [K, Step, Dim]
    bench_noCrash_tensor = torch.FloatTensor(np.array(bench_noCrash))
    if torch.cuda.is_available():
        bench_noCrash_tensor = bench_noCrash_tensor.cuda()
    
    # ----------------------------------------
   
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    episode_rewards, episode_lengths = [], []
    ep_len = 0
    successes = []
    fuzzer = fuzzing()
    seeds_num = 100
    i = 0
    pbar = tqdm.tqdm(total=seeds_num)
    
    # === 循环 1: 初始种群生成 (预热) ===
    while i < seeds_num:
        states = np.array([np.random.uniform(-0.6, -0.4), 0.0])
        state = None
        episode_reward = 0.0
        
        obs = env.reset()
        env.envs[0].unwrapped.state = states
        obs = np.array([states])
        
        sequences = [obs[0]]
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            sequences.append(obs[0])
            episode_reward += reward[0]
            if done:
                break
        
        state = None
        episode_reward_mutate = 0.0
            
        delta_states = np.random.normal(0, 0.05, size=states.shape)
        mutate_states = states + delta_states
        mutate_states[0] = np.clip(mutate_states[0], -0.6, -0.4)
        mutate_states[1] = np.clip(mutate_states[1],0, 0)
            
        obs = env.reset()
        env.envs[0].unwrapped.state = mutate_states
        obs = np.array([mutate_states])

        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            episode_reward_mutate += reward[0]
            if done:
                break
            
        entropy = np.linalg.norm(episode_reward_mutate - episode_reward)
        cvg = fuzzer.state_coverage(sequences)
        fuzzer.further_mutation(states, episode_reward, entropy, cvg, states, 0)
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

    # === 循环 2: Fuzzing 测试 ===
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
    
    while current_time - start_fuzz_time < 3600 * 12 and len(fuzzer.corpus) > 0 and seedcount<1000:
        is_crash = False
        seedcount+=1
        output_obs = []
        temp1_time = time.time()
        
        states = fuzzer.get_pose()
        mutate_states = fuzzer.mutation(states)
        current_gen = fuzzer.current_generation + 1 
        state = None
        episode_reward = 0.0
        
        obs = env.reset()
        env.envs[0].unwrapped.state = mutate_states
        obs = np.array([mutate_states])
        
        sequences = [obs[0]]
        
        # --- Episode Loop ---
        early_terminated = False
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            sequences.append(obs[0])
            if not args.no_render:
                env.render("human")
            episode_reward += reward[0]
            output_obs.append(obs[0])
            
            # [关键修改] Diversity Inference & Early Termination Logic
            # 在指定步数 (CheckPoint) 进行检查，这里是 Hyperparameter.Step
            if (len(output_obs) == Hyperparameter.Step): 
                # 调用新的 K-Voting 预测函数
                ret = predict_siamese.predict_voting(siamese_model, bench_noCrash_tensor, output_obs)
                if ret == 1:
                    print(f'--- Early termination triggered at step {len(output_obs)} ---')
                    early_terminated = True
                    break # 真正的提早终止
                else:
                    print('--- Continue execution (Diverse sequence) ---')
                    
            if done:
                break
        # -------------------

        temp2_time = time.time()
        time_of_env += temp2_time - temp1_time
        
        # 如果提早终止，我们跳过后续的覆盖率更新和Crash检查，或者仅标记为非Crash
        if early_terminated:
             # 提早终止的用例被认为是非多样性的/非Crash的，节省了资源
             # 不进行后续复杂的覆盖率计算或Crash记录
             current_time = time.time()
             print('Episode terminated early. Skipping post-processing.')
             continue 

        cvg = fuzzer.state_coverage(sequences)
        temp3_time = time.time()
        time_of_DynEM += temp3_time - temp2_time
        local_sensitivity = np.abs(episode_reward - fuzzer.current_reward)
        
        final_pos = infos[0]['terminal_observation'][0]

        if final_pos < 0.5: 
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
                timeStamp.write(s + str(current_time) + '\n')
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
                timeStamp.write(s + str(current_time) + '\n')

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
                timeStamp.write(s + str(current_time) + '\n')
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
                timeStamp.write(s + str(current_time) + '\n')

            if cvg < cvg_threshold or episode_reward < fuzzer.current_reward:
                current_pose = copy.deepcopy(mutate_states)
                orig_pose = fuzzer.current_original
                fuzzer.further_mutation(current_pose, episode_reward, local_sensitivity, cvg, orig_pose,current_gen)
        else:
            if episode_reward < fuzzer.current_reward:
                current_pose = copy.deepcopy(mutate_states)
                orig_pose = fuzzer.current_original
                fuzzer.further_mutation(current_pose, episode_reward, local_sensitivity, cvg, orig_pose,current_gen)
        
        log_entry = {
            'state': copy.deepcopy(mutate_states),
            'generation': current_gen,             
            'crashed': is_crash                    
        }
        mutation_log.append(log_entry)
        current_time = time.time()
        time_of_fuzzer += current_time - temp2_time
        print('total reward: ', episode_reward, ', coverage: ', cvg, ', passed time: ', current_time - start_fuzz_time, ', corpus size: ', len(fuzzer.corpus), 'time_of_fuzzer: ', time_of_fuzzer, 'time_of_env: ', time_of_env)
    
    if args.em:
        file_name = os.path.join(result_path, 'crash_EM.pkl')
    else:
        file_name = os.path.join(result_path, 'crash_noEM.pkl')
    with open(file_name, 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(result_path, 'all_run_seeds_0.pkl'), 'wb') as handle:
        pickle.dump(mutation_log, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
                env.envs[0].close()
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