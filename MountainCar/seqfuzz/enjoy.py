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
from tapnet import predict_siamese, Hyperparameter
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="MountainCar-v0")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="../logs")
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
        "--load-best", action="store_true", default=True, help="Load best model instead of last model if available"
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

    # --- 文件夹与日志初始化 ---
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

    # --- Initial Random State ---
    # [随机数修复] 已注释掉硬编码种子，确保 --seed 参数生效
    # np.random.seed(2021) 
    states = np.array([np.random.uniform(-0.6, -0.4), 0.0])
    obs = env.reset()
    env.envs[0].unwrapped.state = states
    obs = np.array([states])

    # --- TapNet Setup ---
    siamese_model = predict_siamese.load_tapnet_mode()
    if torch.cuda.is_available():
        siamese_model.cuda()
    
    weights_path = r'./tapnet/data/weights/tapnet.pkl'
    if os.path.exists(weights_path):
        siamese_model.load_state_dict(torch.load(weights_path))
    else:
        print(f"Warning: Weights not found at {weights_path}")

    siamese_model.eval()
    
    try:
        bench_noCrash = Hyperparameter.bench_noCrash
    except AttributeError:
        bench_noCrash = []

    if len(bench_noCrash) == 0:
        bench_noCrash = torch.zeros((1, Hyperparameter.Step, Hyperparameter.Dimension))
        if torch.cuda.is_available():
            bench_noCrash = bench_noCrash.cuda()
    else:
         bench_noCrash = torch.FloatTensor(np.array(bench_noCrash))
         if torch.cuda.is_available():
             bench_noCrash = bench_noCrash.cuda()
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
    seeds_num = 2000
    i = 0
    pbar = tqdm.tqdm(total=seeds_num)
    
    # === 循环 1: 初始种群生成 ===
    while i < seeds_num:
        # MountainCar Random State
        states = np.array([np.random.uniform(-0.6, -0.4), 0.0])
        state = None
        episode_reward = 0.0
        
        # Reset Env
        obs = env.reset()
        env.envs[0].unwrapped.state = states
        obs = np.array([states])
        
        sequences = [obs[0]]
        
        # 运行第一次 Episode
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            sequences.append(obs[0])
            episode_reward += reward[0]
            if done:
                break
        
        # 判定是否成功 (MountainCar)
        is_success = False
        if done:
            final_pos = -1.2
            if infos and 'terminal_observation' in infos[0]:
                final_pos = infos[0]['terminal_observation'][0]
            elif len(obs) > 0:
                final_pos = obs[0][0] # Fallback
            
            if final_pos >= 0.5:
                is_success = True
        
        if is_success:
            state = None
            episode_reward_mutate = 0.0
            
            # --- Mutation Logic ---
            delta_states = np.random.normal(0, 0.05, size=states.shape)
            mutate_states = states + delta_states
            mutate_states[0] = np.clip(mutate_states[0], -0.6, -0.4)
            mutate_states[1] = np.clip(mutate_states[1], 0, 0) 
            
            # Reset with mutated state
            obs = env.reset()
            env.envs[0].unwrapped.state = mutate_states
            obs = np.array([mutate_states])
            
            print('Mutate states: ', mutate_states)

            for _ in range(args.n_timesteps):
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                episode_reward_mutate += reward[0]
                if done:
                    break
            
            # 计算敏感度 (Entropy)
            entropy = np.abs(episode_reward_mutate - episode_reward)
            
            cvg = fuzzer.state_coverage(sequences)
            fuzzer.further_mutation(states, episode_reward, entropy, cvg, states, 0)
            print(f"Success Seed | Entropy: {entropy:.4f}, Reward: {episode_reward:.2f}, Cvg: {cvg}")
            
            i += 1
            pbar.update(1)
        else:
            pass

    
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
    
    # --- [新增] 创建记录所有Episode观测值的文件 ---
    allObsFile = open(os.path.join(result_path, 'all_episodes_obs.txt'), mode='a')
    # --------------------------------------------

    seedcount = 0
    
    # === 循环 2: Fuzzing Main Loop ===
    while current_time - start_fuzz_time < 3600 * 12 and len(fuzzer.corpus) > 0 and seedcount<5000:
        is_crash = False
        seedcount+=1
        output_obs = []
        temp1_time = time.time()
        
        states = fuzzer.get_pose()
        mutate_states = fuzzer.mutation(states)
        current_gen = fuzzer.current_generation + 1 
        
        state = None
        episode_reward = 0.0
        
        # Reset Env
        obs = env.reset()
        env.envs[0].unwrapped.state = mutate_states
        obs = np.array([mutate_states])
        
        sequences = [obs[0]]
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
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

        # --- [新增] 将当前Episode的所有观测值写入文件 ---
        if len(output_obs) > 0:
            for i in range(len(output_obs)):
                outputStr = ''
                for d in output_obs[i]:
                    outputStr = outputStr + str(d) + ', '
                allObsFile.write(outputStr)
                allObsFile.write('\n')
            allObsFile.write('######') # 分隔符，表示一个Episode结束
            allObsFile.write('\n')
            # 实时刷新缓冲区，防止程序中断丢失数据
            allObsFile.flush()
        # ---------------------------------------------

        temp2_time = time.time()
        time_of_env += temp2_time - temp1_time
        cvg = fuzzer.state_coverage(sequences)
        temp3_time = time.time()
        time_of_DynEM += temp3_time - temp2_time
        local_sensitivity = np.abs(episode_reward - fuzzer.current_reward)
        
        # --- Crash Definition (MountainCar Specific) ---
        final_pos = -1.2
        if infos and 'terminal_observation' in infos[0]:
             final_pos = infos[0]['terminal_observation'][0]
        elif len(obs) > 0:
             final_pos = obs[0][0]

        # 判定条件：如果最终位置小于 0.5，视为失败/Crash
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
        
        # --- 修改位置：添加 crash_time 和 root_seed ---
        current_time_log = time.time()
        log_entry = {
            'state': copy.deepcopy(mutate_states),
            'generation': current_gen,
            'crashed': is_crash,
            'crash_time': current_time_log - start_fuzz_time, # [修改] 添加时间字段
            'root_seed': copy.deepcopy(fuzzer.current_original) # [修改] 添加原始种子信息，方便后续绘图分析
        }
        mutation_log.append(log_entry)
        
        current_time = time.time()
        time_of_fuzzer += current_time - temp2_time
        print('total reward: ', episode_reward, ', coverage: ', cvg, ', passed time: ', current_time - start_fuzz_time, ', corpus size: ', len(fuzzer.corpus), 'time_of_fuzzer: ', time_of_fuzzer, 'time_of_env: ', time_of_env)
    
    # 循环结束后关闭所有文件句柄
    allObsFile.close() # 关闭新增的文件
    successObs.close()
    failObs.close()
    crashF_40.close()
    noCrashF_40.close()
    timeStamp.close()

    if args.em:
        file_name = os.path.join(result_path, 'crash_EM.pkl')
    else:
        file_name = os.path.join(result_path, 'crash_noEM.pkl')
    with open(file_name, 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(result_path, 'all_run_seeds_0.pkl'), 'wb') as handle:
        pickle.dump(mutation_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # --- [新增] 统计导致 Crash 的不同初始种子数量 ---
    unique_root_seeds = set()
    for item in fuzzer.result:
        # 检查 item 是否为字典且包含 root_seed (兼容修改后的 fuzz.py)
        if isinstance(item, dict) and "root_seed" in item:
            seed = item["root_seed"]
            # 如果 seed 是 numpy 数组，转换为 tuple 以便放入 set (因为 numpy array 不可哈希)
            if isinstance(seed, np.ndarray):
                seed = tuple(seed.tolist())
            elif isinstance(seed, list):
                seed = tuple(seed)
            unique_root_seeds.add(seed)
    
    print(f"Total Crashes Found: {len(fuzzer.result)}")
    print(f"Unique Root Seeds Causing Crash: {len(unique_root_seeds)}")
    # ------------------------------------------------

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")
    
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