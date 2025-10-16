import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch

import sys
sys.path.append('../../methods/src/')
from bw_executor import BipedalWalkerExecutor
from tt_executor import TaxiExecutor
from ll_executor import LunarLanderExecutor
from mdpfuzz import Fuzzer


'''
Python script that launches Fuzzer, MDPFuzz or RT for one of the RL use-cases (bw, ll or tt).
Args:
    1. Path for logging.
    2. A positive integer "i". The method is mapped with i // 5, and the random seed with i % 5.
    3. The RL key (bw, ll or tt).
'''

EXPERIMENT_SEEDS = [2021, 42, 2023, 20, 0]
RL_KEYS = ['bw', 'll', 'tt']
RL_NAMES = ['Bipedal Walker', 'Lunar Lander', 'Taxi']


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_budget = 5000
    init_budget = 1000
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


    if rl_index == 0:
        executor = BipedalWalkerExecutor(300, 0)
    elif rl_index == 1:
        executor = LunarLanderExecutor(1000, 0)
    else:
        executor = TaxiExecutor(0, 0)


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
            test_budget=test_budget,
            saving_path=path,
            local_sensitivity=True,
            save_logs_only=True,
            exp_name=RL_NAMES[rl_index])
    else:
        fuzzer.fuzzing(
            n=init_budget,
            policy=policy,
            test_budget=test_budget,
            saving_path=path,
            local_sensitivity=True,
            save_logs_only=True,
            exp_name=RL_NAMES[rl_index])