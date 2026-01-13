import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import argparse
import sys
sys.path.append('../../methods/src/')
from bw_executor import BipedalWalkerExecutor
from fuzz.mdpfuzz import Fuzzer
from datetime import datetime

EXPERIMENT_SEEDS = [0,1,42,723,1022]
RL_KEYS = ['bw']
RL_NAMES = ['Bipedal Walker']


if __name__ == '__main__':
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- 脚本开始运行时间: {start_time_str} ---")
    torch.set_num_threads(1)
    
    test_budget = 7000
    test_budget_in_seconds = 3600
    init_budget = 1000
    k = 10
    tau = 0.01
    gamma = 0.01

    if len(sys.argv) < 4:
        print("Usage: python test_rl.py <path> <i> <rl_key> [options]")
        sys.exit(1)

    log_path_arg = sys.argv[1]
    i_arg = sys.argv[2]
    rl_key_arg = sys.argv[3]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-data", action="store_true", default=False, help="Save TodyNet training data")
    parser.add_argument("--save-transitions", action="store_true", default=False, help="Save RL transitions")
    parser.add_argument("--window-size", type=int, default=25, help="Sliding window size")
    args, unknown = parser.parse_known_args(sys.argv[4:])

    if os.path.isdir(log_path_arg) and not log_path_arg.endswith('/'):
        log_path_arg += '/'

    method_names = ['fuzzer', 'mdpfuzz', 'rt']

    method_index = int(i_arg) // len(EXPERIMENT_SEEDS)
    method = method_names[method_index]

    seed_index = int(i_arg) % len(EXPERIMENT_SEEDS)
    seed = EXPERIMENT_SEEDS[seed_index]

    rl = rl_key_arg
    assert rl in RL_KEYS
    rl_index = RL_KEYS.index(rl)

    result_path = log_path_arg + rl + '/' + method
    path = '{}_{}_{}_{}_{}'.format(result_path, k, tau, gamma, seed)
    print(f"Log Path: {path}")
    print(f"Data Collection: SaveData={args.save_data}, SaveTrans={args.save_transitions}")

    executor = BipedalWalkerExecutor(300, 0)
    
    policy = executor.load_policy()
    fuzzer = Fuzzer(random_seed=seed, k=k, tau=tau, gamma=gamma, executor=executor)

    fuzz_kwargs = {
        'n': init_budget,
        'policy': policy,
        'test_budget_in_seconds': test_budget_in_seconds,
        'saving_path': path,
        'local_sensitivity': True,
        'save_logs_only': True,
        'exp_name': RL_NAMES[rl_index],
        'save_data': args.save_data,
        'save_transitions': args.save_transitions,
        'window_size': args.window_size
    }

    if method == 'rt':
        fuzzer.random_testing(
            n=test_budget,
            policy=policy,
            path=path,
            exp_name=RL_NAMES[rl_index])
    elif method == 'fuzzer':
        fuzzer.fuzzing_no_coverage(**fuzz_kwargs)
    else:
        fuzzer.fuzzing(**fuzz_kwargs)

    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- 脚本结束运行时间: {end_time_str} ---")
    duration = end_time - start_time
    print(f"--- 总计运行时间: {duration} ---")