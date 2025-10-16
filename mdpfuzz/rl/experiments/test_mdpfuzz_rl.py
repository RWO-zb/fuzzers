import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch

import sys
sys.path.append('../../methods/src/')
from mdpfuzz import Fuzzer
from bw_executor import BipedalWalkerExecutor
from tt_executor import TaxiExecutor
from ll_executor import LunarLanderExecutor

'''
Python script that executes MDPFuzz for given parameters (or reads the latter in the related file).
Args:
    1. The rl "key" (bw, ll or tt).
    2. Either k, tau, gamma and a random seed or an integer which refers the number of the line in the parameters.txt.
    3. Path for logging. It is appended by "k_tau_gamma_seed".
'''

RL_KEYS = ['bw', 'll', 'tt']
RL_NAMES = ['Bipedal Walker', 'Lunar Lander', 'Taxi']

if __name__ == '__main__':
    torch.set_num_threads(1)
    test_budget = 5000
    init_budget = 1000

    tmp = sys.argv[1:]

    # reads parameters and path in the parameters.txt if not given
    if len(tmp) == 6:
        args = tmp
    else:
        assert len(tmp) == 3
        index = int(tmp[1])
        with open('../../parameters.txt', 'r') as f:
            lines = f.readlines()
        parameters = lines[index].strip().split(' ')
        assert len(parameters) == 4
        args = []
        for p in parameters:
            args.append(p)
        args.append(tmp[2])
        args.append(tmp[0])

    k = int(args[0])
    tau = float(args[1])
    gamma = float(args[2])
    seed = int(args[3])


    print('k', 'tau', 'gamma', 'seed')
    print(k, tau, gamma, seed)

    path = '{}_{}_{}_{}_{}'.format(args[4], k, tau, gamma, seed)
    print(path)

    rl = args[5]
    assert rl in RL_KEYS
    rl_index = RL_KEYS.index(rl)

    if rl_index == 0:
        executor = BipedalWalkerExecutor(300, 0)
    elif rl_index == 1:
        executor = LunarLanderExecutor(1000, 0)
    else:
        executor = TaxiExecutor(0, 0)

    policy = executor.load_policy()

    fuzzer = Fuzzer(random_seed=seed, k=k, tau=tau, gamma=gamma, executor=executor)
    fuzzer.fuzzing(
        n=init_budget,
        policy=policy,
        test_budget=test_budget,
        saving_path=path,
        local_sensitivity=True,
        save_logs_only=True,
        exp_name=RL_NAMES[rl_index])