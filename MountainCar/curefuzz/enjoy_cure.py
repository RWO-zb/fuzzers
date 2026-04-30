import argparse, importlib, os, sys, time, copy, tqdm, pickle
import gymnasium as gym 
import yaml
import numpy as np
import torch as th
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper
import utils.import_envs
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from fuzz.cure_fuzz import CureFuzz
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="MountainCar-v0")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="../logs")
    parser.add_argument("--algo", help="RL Algorithm", default="dqn", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=200, type=int)
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
    parser.add_argument("--guide", action="store_true", default=False)
    parser.add_argument("--intrinsic", help="Threshold for intrinsic reward", default=10, type=int)
    parser.add_argument("--entropy", help="Threshold for reward", default=10, type=int)
    parser.add_argument("--seed_number", help="Number of seeds", default=2000, type=int)
    args = parser.parse_args()
    
    now_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    result_folder = f"{now_str}_seed_{args.seed}"
    if not os.path.exists('./results'):
        os.mkdir('./results')
    result_path = './results/' + result_folder + '/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    log_file_path = os.path.join(result_path, 'cure_fuzz.txt')
    f = open(log_file_path, 'w', buffering=1)
    sys.stdout = f
    sys.stderr = f 
    
    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    intrins_theta = args.intrinsic
    entropy_theta = args.entropy
    
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
    
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic
    episode_rewards, episode_lengths = [], []
    successes = []
    fuzzer = CureFuzz()
    seeds_num = args.seed_number
    i = 0
    pbar = tqdm.tqdm(total=seeds_num)
    start_corpus_time = time.time()
    
    while i < seeds_num and (time.time() - start_corpus_time) <= (3600*2):
        
        pos = np.random.uniform(-0.6, -0.4)
        vel = 0.0
        states = np.array([pos, vel])

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
                # print(infos[0]['terminal_observation'][0])
                break
        
        final_state = sequences[-1] if len(sequences) > 1 else sequences[0]

        state = None
        episode_reward_mutate = 0.0

        mutate_states = fuzzer.mutation(states)

        obs = env.reset()
        env.envs[0].unwrapped.state = mutate_states
        obs = np.array([mutate_states])
        
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            episode_reward_mutate += reward[0]
            if done:
                    break
        
        entropy = np.linalg.norm(np.asarray(final_state) - np.asarray(obs[0]))
        intrinsic_reward = fuzzer.train_rnd(sequences)    
        fuzzer.further_mutation(states, episode_reward, entropy, intrinsic_reward, final_state, states)  
        i += 1
        pbar.update(1)
        
    fuzzer.count = [5] * len(fuzzer.corpus)
    fuzzer.original = copy.deepcopy(fuzzer.corpus)

    all_traces = []

    start_fuzz_time = time.time()
    current_time = time.time()
    pbar1 = tqdm.tqdm(total=seeds_num) 
    seedcount = 0
    
    fuzz_selection_log = []
    all_obs_sequences = [] 

    while current_time - start_fuzz_time < (3600 * 12) and len(fuzzer.corpus) > 0 :
        seedcount += 1

        selected_info = fuzzer.get_pose()
        states = selected_info['seed_state']
        current_mutation_depth = selected_info['depth'] 

        mutate_states = fuzzer.mutation(states)
        state = None
        episode_reward = 0.0
        
        obs = env.reset()
        env.envs[0].unwrapped.state = mutate_states
        obs = np.array([mutate_states])

        sequences = [obs[0]]
        
        trace_record = {
            "steps": [],
            "positions": [],
            "velocities": [],
            "action_gaps": [],
            "q_values": [],
            "actions": [],
            "rewards": [],
            "start_state": mutate_states
        }
        
        reached_goal = False
        is_crash_condition = False 

        for step_idx in range(args.n_timesteps):
            obs_tensor, _ = model.q_net.obs_to_tensor(obs)
            with th.no_grad():
                q_values = model.q_net(obs_tensor).cpu().numpy()[0]
            
            sorted_q = np.sort(q_values)
            gap = sorted_q[-1] - sorted_q[-2]
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            
            trace_record["steps"].append(step_idx)
            trace_record["positions"].append(obs[0][0])
            trace_record["velocities"].append(obs[0][1])
            trace_record["action_gaps"].append(gap)
            trace_record["q_values"].append(q_values)
            trace_record["actions"].append(action[0])

            obs, reward, done, infos = env.step(action)
            trace_record["rewards"].append(reward[0]) 
            
            episode_reward += reward[0]

            if done:
                terminal_obs = infos[0]['terminal_observation']
                sequences.append(terminal_obs)
                if terminal_obs[0] < 0.5:
                    is_crash_condition = True
                break
            else:
                sequences.append(obs[0])
        
        all_obs_sequences.append(sequences)

        trace_record["did_crash"] = is_crash_condition
        trace_record["total_reward"] = episode_reward
        
        all_traces.append(trace_record)

        intrinsic_reward = fuzzer.train_rnd(sequences)
        current_final_state = fuzzer.current_final_state if fuzzer.current_final_state is not None else obs[0]
        entropy = np.linalg.norm(np.asarray(obs[0]) - np.asarray(current_final_state))

        did_crash = False
        crash_time_point = None 

        if is_crash_condition:
            pbar1.update(1)
            fuzzer.add_crash(mutate_states)
            print('Found: ', len(fuzzer.result))
            did_crash = True
            
            crash_time_point = time.time() - start_fuzz_time
            print(f"  -> Crash found at {crash_time_point:.2f} seconds")
            
        elif args.guide:
            if intrinsic_reward > intrins_theta or episode_reward < fuzzer.current_reward or entropy > entropy_theta:
                current_pose = copy.deepcopy(mutate_states)
                orig_pose = fuzzer.current_original
                fuzzer.further_mutation(current_pose, episode_reward,  entropy, intrinsic_reward, final_state, orig_pose)
        else:
            if episode_reward < fuzzer.current_reward or entropy > entropy_theta:
                current_pose = copy.deepcopy(mutate_states)
                orig_pose = fuzzer.current_original
                fuzzer.further_mutation(current_pose, episode_reward,  entropy, intrinsic_reward, final_state, orig_pose)
        
        log_entry = {
            'seed_state': selected_info['seed_state'], 
            'mutate_state': mutate_states,          
            'parent_depth': current_mutation_depth, 
            'did_crash': did_crash,
            'crash_time': crash_time_point
        }
        fuzz_selection_log.append(log_entry)
        
        print(f'Total seeds tested: { seedcount}, Crashes found: {len(fuzzer.result)}, Corpus size: {len(fuzzer.corpus)}')
        current_time = time.time()
        
    if args.guide:
        file_name = os.path.join(result_path, 'cure_crash.pkl')
    else:
        file_name = os.path.join(result_path, 'ablated_crash.pkl')
    with open(file_name, 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    log_file_name = os.path.join(result_path, 'selection_log.pkl')
    with open(log_file_name, 'wb') as handle:
        pickle.dump(fuzz_selection_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Selection log saved to {log_file_name}")

    obs_seq_file_name = os.path.join(result_path, 'obs_sequences.pkl')
    with open(obs_seq_file_name, 'wb') as handle:
        pickle.dump(all_obs_sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Observation sequences saved to {obs_seq_file_name}")

    trace_data_file = os.path.join(result_path, 'trace_data.pkl')
    with open(trace_data_file, 'wb') as f:
        pickle.dump(all_traces, f)

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