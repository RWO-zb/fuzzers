import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch

import sys
sys.path.append('../../methods/src/')
from bw_executor import BipedalWalkerExecutor
from fuzz.mdpfuzz import Fuzzer
from datetime import datetime

'''
Python script that launches Fuzzer, MDPFuzz or RT for one of the RL use-cases (bw, ll or tt).
Args:
    1. Path for logging.
    2. A positive integer "i". The method is mapped with i // 5, and the random seed with i % 5.
    3. The RL key (bw, ll or tt).
'''

EXPERIMENT_SEEDS = [0,1,42,723,1022]
RL_KEYS = ['bw']
RL_NAMES = ['Bipedal Walker']


if __name__ == '__main__':
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- 脚本开始运行时间: {start_time_str} ---")
    torch.set_num_threads(1)
    test_budget = 500
    test_budget_in_seconds=180
    init_budget = 10
    k = 10
    tau = 0.01
    gamma = 0.01


    args = sys.argv[1:]
    assert len(args) == 3

    path = args[0]

    if os.path.isdir(path) and not path.endswith('/'):
        path += '/'

    method_names = ['fuzzer', 'mdpfuzz', 'rt']

    method_index = int(args[1]) // len(EXPERIMENT_SEEDS)
    method = method_names[method_index]


    seed_index = int(args[1]) % len(EXPERIMENT_SEEDS)
    seed = EXPERIMENT_SEEDS[seed_index]


    rl = args[2]
    assert rl in RL_KEYS
    rl_index = RL_KEYS.index(rl)


    result_path = path + rl + '/' + method
    path = '{}_{}_{}_{}_{}'.format(result_path, k, tau, gamma, seed)
    print(path)

    executor = BipedalWalkerExecutor(300, 0)
    
    policy = executor.load_policy()
    fuzzer = Fuzzer(random_seed=seed, k=k, tau=tau, gamma=gamma, executor=executor)

    if method == 'rt':
        fuzzer.random_testing(
            n=test_budget,
            policy=policy,
            path=path,
            exp_name=RL_NAMES[rl_index])
    elif method == 'fuzzer':
        fuzzer.fuzzing_no_coverage(
            n=init_budget,
            policy=policy,
            test_budget_in_seconds=test_budget_in_seconds,
            saving_path=path,
            local_sensitivity=True,
            save_logs_only=True,
            exp_name=RL_NAMES[rl_index])
    else:
        fuzzer.fuzzing(
            n=init_budget,
            policy=policy,
            test_budget_in_seconds=test_budget_in_seconds,
            saving_path=path,
            local_sensitivity=True,
            save_logs_only=True,
            exp_name=RL_NAMES[rl_index])
    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- 脚本结束运行时间: {end_time_str} ---")
    duration = end_time - start_time
    print(f"--- 总计运行时间: {duration} ---")
