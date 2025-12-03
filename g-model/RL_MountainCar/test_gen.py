import argparse, importlib, os, sys, time, copy, tqdm, pickle, yaml
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecVideoRecorder
import utils.import_envs
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
import json, random, pickle, math
from datetime import datetime

from interfaces import normalize_data, Memory, Density, compute_sensitivity, case_clip, compute_novelty, Grid
from diffusion import Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="MountainCar-v0")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--algo", help="RL Algorithm", default="dqn", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=200, type=int) 
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=8, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
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


    ######################## parameters for generative testing ############################################
    parser.add_argument("--method", help="select the guidance for testing", default="generative+novelty", type=str, required=False)
    parser.add_argument("--hour", help="test time", default=12, type=int)
    parser.add_argument("--step", help="number of normal cases at each training step", default=50, type=int)
    parser.add_argument("--grid", help="state abstraction granularity", default=5, type=int) 
    args = parser.parse_args()

    result_folder_name = f"MC_{args.method}_{args.step}_seed_{args.seed}"
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

    ##################################################################################################

    case_dimension = 2 
    diffusion_model = Diffusion(batch_size = 1, epoch = 100, data_size = case_dimension, training_step_per_spoch = 25, num_diffusion_step = 25)
    diffusion_model.setup()
    memory_model = Memory(size = 100)
    density_model = Density()


    ################################### nvovelty computation ########################################
    min_obs = np.array([-1.2, -0.07])
    max_obs = np.array([0.6, 0.07])
    
    novelty_grid = Grid(min_obs, max_obs, args.grid)
    novelty_test_grid = Grid(min_obs, max_obs, args.grid)
    novelty_dict = dict()
    novelty_test_dict = dict()

    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    episode_rewards, episode_lengths = [], []
    ep_len = 0

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

    all_test_cases_log = []
    all_trajectories = [] # 用于保存主循环中的轨迹

    def get_random_mc_state():
        pos = np.random.uniform(-0.6, -0.4) 
        vel = 0
        return np.array([pos, vel])

    # --- 阶段 1：严格遵循论文的初始化预热 (Strict Initialization) ---
    print("--- Stage 1: Initialization (Warm-up) ---")
    initial_collection_count = 1000 # 论文要求：采样 1000 个正常用例
    
    for pre_step in tqdm.tqdm(range(initial_collection_count), desc="Initial Random Sampling"):
        # 仅生成随机初始状态，不执行环境交互，不计算 metrics
        normal_case = get_random_mc_state()
        normal_case_list.append(normal_case)

    # --- 阶段 2：预训练扩散模型 (Pre-training) ---
    if len(normal_case_list) > 0:
        print(f"--- Pre-training Diffusion Model with {len(normal_case_list)} samples ---")
        normal_case_list = np.array(normal_case_list)
        
        # 论文要求：在第一阶段，扩散模型仅捕捉正常分布，不涉及 metrics。
        # 强制 metrics=None 且 method='generative'
        diffusion_model.train(normal_case_list, None, 'generative')
        
        # 清空数据，为正式测试循环做准备
        normal_case_list = []
        metric_list = []
        memory_model.clear()

    # --- 阶段 3：主循环 ---
    print("--- Stage 3: Main Testing Loop ---")
    start_time = time.time()
    current_time = time.time()
    
    while current_time - start_time < 3600 * 0.05: # 使用参数 args.hour 控制时长

        if cur_step > 0 and cur_step % args.step == 0:
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

            for _ in range(args.step): # 使用 step 大小进行验证
                failure_flag = False
                state = None
                
                # --- 生成测试用例 ---
                test_case = diffusion_model.generate()[0] # diffusion返回 (1,2)，取[0]得 (2,)
                
                # --- 设置环境状态 ---
                obs = env.reset()
                env.envs[0].unwrapped.state = test_case
                obs = np.array([test_case])

                sequences = [obs[0]]
                episode_reward = 0.0

                for _ in range(args.n_timesteps):
                    action, state = model.predict(obs, state=state, deterministic=deterministic)
                    obs, reward, done, infos = env.step(action)
                    
                    # 记录 OBS 序列
                    if done and "terminal_observation" in infos[0]:
                        sequences.append(infos[0]["terminal_observation"])
                    else:
                        sequences.append(obs[0])
                        
                    episode_reward += reward[0]
                    if done:
                        break
                
                # 保存 Generative 阶段的轨迹
                all_trajectories.append(sequences)

                # --- Crash 定义修改 ---
                if "terminal_observation" in infos[0]:
                    is_crash = (infos[0]['terminal_observation'][0] < 0.5)
                else:
                    is_crash = (obs[0][0] < 0.5)
                
                # 计算时间戳
                elapsed_time = time.time() - start_time

                all_test_cases_log.append({
                    "input": test_case.tolist(), 
                    "is_crash": bool(is_crash),
                    "source": "generative",
                    "step": cur_step,
                    "timestamp": elapsed_time
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
                        print(f"Crash Found (Diff)! Reward: {episode_reward:.2f}, State: {save_case}")
                        diffusion_failure_count.append([regular_time, failure_by_diffusion, save_case])
                else:
                    wins += 1  

                abstract_id = novelty_grid.state_abstract(np.array([sequences[-1]]))[0]
                if abstract_id in novelty_test_dict.keys():
                    novelty_dict[abstract_id] += 1
                else:
                    novelty_dict[abstract_id] = 1
                novelty = novelty_dict[abstract_id]
                # 修复 OverflowError
                try:
                    norm_novelty = 1 / (math.e ** (novelty - 1))
                except OverflowError:
                    norm_novelty = 0.0

                normal_case_list.append(test_case)
                metric_list.append([0, 0, 0, norm_novelty])
                memory_model.append(test_case, 0, 0, 0, novelty)

                if failure_flag:
                    diffusion_failure_clusters.append(abstract_id)

                information_list.append([sequences[-1].tolist(), failure_flag, abstract_id, norm_novelty])
        else:
            # --- 随机生成逻辑 ---
            state = None
            normal_case = get_random_mc_state()
            
            obs = env.reset()
            env.envs[0].unwrapped.state = normal_case
            obs = np.array([normal_case])
            
            sequences = [obs[0]]
            episode_reward = 0.0
            
            for _ in range(args.n_timesteps):
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                
                # 记录 OBS 序列
                if done and "terminal_observation" in infos[0]:
                    sequences.append(infos[0]["terminal_observation"])
                else:
                    sequences.append(obs[0])
                    
                episode_reward += reward[0]
                if done:
                    break
            
            # 保存 Random 阶段的轨迹
            all_trajectories.append(sequences)
            
            if "terminal_observation" in infos[0]:
                is_crash = (infos[0]['terminal_observation'][0] < 0.5)
            else:
                is_crash = (obs[0][0] < 0.5)

            # 计算时间戳
            elapsed_time = time.time() - start_time

            all_test_cases_log.append({
                "input": normal_case.tolist(), 
                "is_crash": bool(is_crash), 
                "source": "random",
                "step": cur_step,
                "timestamp": elapsed_time
            })
            
            if is_crash:
                 print(f"Crash Found (Rand)! Reward: {episode_reward:.2f}, State: {normal_case.tolist()}")

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
                # 修复 OverflowError
                try:
                    norm_novelty = 1 / (math.e ** (novelty - 1))
                except OverflowError:
                    norm_novelty = 0.0

            metric_list.append([norm_density, norm_sensitivity, norm_performance, norm_novelty])
            memory_model.append(normal_case, density, sensitivity, performance, novelty)

            if cur_step % 10 == 0:
                print(f"Step: {cur_step}, Reward: {episode_reward:.2f}, Case: {normal_case}")

        cur_step += 1
        current_time = time.time()

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
        
    # 保存所有轨迹 (PKL)
    traj_pkl_filename = os.path.join(result_path, 'all_trajectories.pkl')
    with open(traj_pkl_filename, 'wb') as f:
        pickle.dump(all_trajectories, f)

if __name__ == '__main__':  
    start_time = datetime.now()
    print(f"--- start time: {start_time} ---")
    main()
    end_time = datetime.now()
    print(f"--- total time: {end_time - start_time} ---")