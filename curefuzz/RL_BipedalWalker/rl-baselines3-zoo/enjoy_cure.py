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
from datetime import datetime
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from fuzz.cure_fuzz import CureFuzz

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="../rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
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
    parser.add_argument("--seed_number", help="Number of seeds", default=2, type=int)
    
    args = parser.parse_args()
    
    now_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    result_folder = f"{now_str}_seed_{args.seed}"
    result_path = './results/' + result_folder + '/'
    os.makedirs(result_path, exist_ok=True)
    
    log_file_path = os.path.join(result_path, 'cure_fuzz.txt')
    f = open(log_file_path, 'w', buffering=1)
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

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

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
    
    pbar = tqdm.tqdm(total=seeds_num)
    start_corpus_time = time.time()
    i = 0
    
    # Corpus Generation Loop
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
    pbar1 = tqdm.tqdm(total=seeds_num)
    seedcount = 0
    fuzz_selection_log = []

    # Fuzzing Loop
    while current_time - start_fuzz_time < (3600 * 12) and len(fuzzer.corpus) > 0:
        seedcount += 1
        selected_info = fuzzer.get_pose()
        states = selected_info['seed_state']
        current_mutation_depth = selected_info['depth']

        mutate_states = fuzzer.mutation(states)
        state = None
        episode_reward = 0.0
        obs = env.reset(mutate_states)
        sequences = [obs[0]]
        
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            sequences.append(obs[0])
            episode_reward += reward[0]
            if done:
                break
        
        intrinsic_reward = fuzzer.train_rnd(sequences)
        entropy = np.linalg.norm(np.asarray(obs[0]) - np.asarray(fuzzer.final_state))
        
        did_crash = False
        if done or episode_reward < 10:
            pbar1.update(1)
            fuzzer.add_crash(mutate_states)
            print('Found: ', len(fuzzer.result))
            did_crash = True
        else:
            condition = False
            if args.guide:
                condition = intrinsic_reward > intrins_theta or episode_reward < fuzzer.current_reward or entropy > entropy_theta
            else:
                condition = episode_reward < fuzzer.current_reward or entropy > entropy_theta
            
            if condition:
                fuzzer.further_mutation(copy.deepcopy(mutate_states), episode_reward, entropy, intrinsic_reward, final_state, fuzzer.current_original)
        
        fuzz_selection_log.append({
            'seed_state': selected_info['seed_state'],
            'mutate_state': mutate_states,
            'parent_depth': current_mutation_depth,
            'did_crash': did_crash,
            'elapsed_time': time.time() - start_fuzz_time
        })
        
        print(f'Total seeds tested: {seedcount}, Crashes found: {len(fuzzer.result)}')
        current_time = time.time()

    crash_file = 'cure_crash.pkl' if args.guide else 'ablated_crash.pkl'
    with open(os.path.join(result_path, crash_file), 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    log_file_name = os.path.join(result_path, 'selection_log.pkl')
    with open(log_file_name, 'wb') as handle:
        pickle.dump(fuzz_selection_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Selection log saved to {log_file_name}")

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
    print(f"--- start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    main()
    end_time = datetime.now()
    print(f"--- end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"--- total time: {end_time - start_time} ---")