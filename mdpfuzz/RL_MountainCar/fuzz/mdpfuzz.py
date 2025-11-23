import os
import time
import copy
import json
import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any

# 包内导入处理
if __package__ is None or __package__ == '':
    from gmm import CoverageModel
    from logger import FuzzerLogger
    from executor import Executor
    from pool import Pool, IndexedPool, LightPool
else:
    from .gmm import CoverageModel
    from .logger import FuzzerLogger
    from .executor import Executor
    from .pool import Pool, IndexedPool, LightPool

class Fuzzer():
    def __init__(self, random_seed: int, k: int, tau: float, gamma: float, executor: Executor) -> None:
        self.k = k
        self.tau = tau
        self.gamma = gamma
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        self.coverage_model = CoverageModel(random_seed, k, gamma)
        self.evaluated_solutions = []
        self.executor = executor
        self.sim_steps = self.executor.sim_steps
        self.env_seed = self.executor.env_seed
        self._set_config()

    def _set_config(self):
        self.config = {
            'k': self.k,
            'gamma': self.gamma,
            'tau': self.tau,
            'random_seed': self.random_seed,
            'random_state': self.rng.bit_generator.state,
            'env_seed': self.env_seed,
            'sim_steps': self.sim_steps,
            'name': 'MDPFuzz',
            'use_case': type(self.executor).__name__
        }

    def _concatenate_state_sequence(self, state_sequence: np.ndarray) -> np.ndarray:
        data_concat = []
        for i in range(len(state_sequence) - 1):
            data_concat.append(np.hstack([state_sequence[i], state_sequence[i+1]]))
        return np.array(data_concat)

    def sampling(self, n: int = 1) -> List[np.ndarray]:
        if n == 1:
            return self.executor.generate_input(self.rng)
        else:
            return self.executor.generate_inputs(self.rng, n=n)

    def mutate(self, state: np.ndarray, **kwargs):
        return self.executor.mutate(state, self.rng, **kwargs)

    def mutate_validate(self, state: np.ndarray, **kwargs):
        attempts = 1
        while attempts < 100:
            mutate_states = self.mutate(state, **kwargs)
            tmp = mutate_states.tolist()
            if not (tmp in self.evaluated_solutions):
                self.evaluated_solutions.append(tmp)
                break
            else:
                attempts += 1
        return mutate_states

    def mdp(self, state: np.ndarray, policy: Any = None) -> Tuple[float, bool, np.ndarray, float]:
        episode_reward, done, obs_seq, exec_time = self.executor.execute_policy(state, policy)
        return episode_reward, done, obs_seq, exec_time

    def sentivity(self, state: np.ndarray, acc_reward: float = None, policy: Any = None, generation: int = None ,**kwargs):
        perturbed_state = self.mutate_validate(state, **kwargs)
        perturbation = np.linalg.norm(state - perturbed_state)

        if acc_reward is None:
            acc_reward, crash, state_sequence, exec_time = self.mdp(state, policy)
        else:
            state_sequence = []
            crash = None
            exec_time = None

        acc_reward_perturbed, crash_perturbed, state_sequence_perturbed, exec_time_perturbed = self.mdp(perturbed_state, policy)
        
        if self.logger is not None:
            episode_length = len(state_sequence_perturbed)
            self.logger.log(
                input=perturbed_state,
                oracle=crash_perturbed,
                reward=acc_reward_perturbed,
                episode_length=episode_length,
                Generation=generation,
                test_exec_time=exec_time_perturbed,
                run_time=time.time()
            )

        sensitivity = np.abs(acc_reward - acc_reward_perturbed) / perturbation
        return sensitivity, acc_reward, crash, state_sequence, exec_time

    def local_sensitivity(self, state: np.ndarray, state_mutate: np.ndarray, state_reward: float, state_mutate_reward: float):
        perturbation = np.linalg.norm(state - state_mutate)
        return np.abs(state_reward - state_mutate_reward) / perturbation

    def initialize_coverage_model(self, **kwargs) -> int:
        exec_counter = kwargs.get('exec_counter', 0)
        state_sequence = kwargs.pop('state_sequence', None)
        if state_sequence is None:
            policy = kwargs.get('policy', None)
            random_input = kwargs.get('input', self.sampling())
            reward, crash, state_sequence, exec_time = self.mdp(random_input, policy)
            exec_counter += 1
            if self.logger is not None:
                self.logger.log(
                    input=random_input,
                    oracle=crash,
                    reward=reward,
                    episode_length=len(state_sequence),
                    Generation=0,
                    test_exec_time=exec_time,
                    run_time=time.time()
                    )

        if len(state_sequence) < self.k + 1:
            kwargs['exec_counter'] = exec_counter
            return self.initialize_coverage_model(**kwargs)
        else:
            self.coverage_model.initialize(state_sequence)
        return exec_counter

    def fuzzing(self, n: int, policy: Any = None, **kwargs):
        if kwargs.get('exp_name', None) is not None:
            self.config['use_case'] = kwargs['exp_name']
        path = kwargs.get('saving_path', None)
        if path is not None:
            self.logger = FuzzerLogger(path + '_logs.txt')
            self.logger.write_columns()
        else:
            self.logger = None

        local_sensitivity = kwargs.get('local_sensitivity', False)
        initial_inputs = self.sampling(n)
        self.config['init_budget'] = n
        pool = IndexedPool(is_integer=np.issubdtype(initial_inputs.dtype, np.integer))

        num_initial_executions = self.initialize_coverage_model(policy=policy)
        self.config['num_initial_executions'] = num_initial_executions
        
        pbar = tqdm.tqdm(total=n)
        for state in initial_inputs:
            sensitivity, acc_reward, oracle, state_sequence, exec_time = self.sentivity(state, policy=policy, generation=0, **kwargs)
            state_sequence_conc = self._concatenate_state_sequence(state_sequence)
            t0 = time.time()
            coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
            coverage_time = time.time() - t0
            pool.add(state, acc_reward, coverage, sensitivity, oracle, generation=0)

            if self.logger is not None:
                self.logger.log(
                    input=state,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=len(state_sequence),
                    sensitivity=sensitivity,
                    coverage=coverage,
                    Generation=0,
                    test_exec_time=exec_time,
                    coverage_time=coverage_time,
                    run_time=time.time()
                )
            if oracle:
                pool.add_crash(state)
            pbar.update(1)
        pbar.close()

        test_budget = kwargs.get('test_budget', None)
        test_budget -=  (2 * n) + num_initial_executions
        self.config['test_budget'] = test_budget
        
        pbar = tqdm.tqdm(total=test_budget)
        num_iterations = 0

        while num_iterations < test_budget:
            input, acc_reward_input, generation = pool.select(self.rng)
            new_generation = generation + 1
            mutant = self.mutate_validate(input, **kwargs)
            acc_reward_mutant, oracle, state_sequence, exec_time = self.mdp(mutant, policy)
            state_sequence_conc = self._concatenate_state_sequence(state_sequence)
            t0 = time.time()
            coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
            coverage_time = time.time() - t0
            
            sensitivity = None
            if oracle:
                pool.add_crash(mutant)
            elif (acc_reward_mutant < acc_reward_input) or (coverage < self.tau):
                if local_sensitivity:
                    sensitivity = self.local_sensitivity(input, mutant, acc_reward_input, acc_reward_mutant)
                else:
                    sensitivity, _, _, _, _ = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy, generation=new_generation, **kwargs)
                pool.add(mutant, acc_reward_mutant, coverage, sensitivity, oracle, generation=new_generation)

            if self.logger is not None:
                self.logger.log(
                    input=mutant,
                    oracle=oracle,
                    reward=acc_reward_mutant,
                    episode_length=len(state_sequence),
                    sensitivity=sensitivity,
                    coverage=coverage,
                    Generation=new_generation,
                    test_exec_time=exec_time,
                    coverage_time=coverage_time,
                    run_time=time.time()
                )
            
            num_iterations += 1
            pbar.update(1)
        
        pbar.close()

        if path is not None:
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')

    def fuzzing_no_coverage(self, n: int, policy: Any = None, **kwargs):
        '''Works similarly as fuzzing but coverages are not computed.'''
        if kwargs.get('exp_name', None) is not None:
            self.config['use_case'] = kwargs['exp_name']
        self.config['name'] = 'Fuzzer_No_Cov'
        
        path = kwargs.get('saving_path', None)
        if path is not None:
            self.logger = FuzzerLogger(path + '_logs.txt')
            self.logger.write_columns()
        else:
            self.logger = None

        local_sensitivity = kwargs.get('local_sensitivity', False)
        initial_inputs = self.sampling(n)
        self.config['init_budget'] = n
        pool = IndexedPool(is_integer=np.issubdtype(initial_inputs.dtype, np.integer))
        
        pbar = tqdm.tqdm(total=n)
        for state in initial_inputs:
            sensitivity, acc_reward, oracle, state_sequence, exec_time = self.sentivity(state, policy=policy, generation=0, **kwargs)
            # 注意：Coverage 传入 0
            pool.add(state, acc_reward, 0, sensitivity, oracle, generation=0)

            if self.logger is not None:
                self.logger.log(
                    input=state,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=len(state_sequence),
                    sensitivity=sensitivity,
                    Generation=0,
                    test_exec_time=exec_time,
                    run_time=time.time()
                )
            if oracle:
                pool.add_crash(state)
            pbar.update(1)
        pbar.close()

        test_budget = kwargs.get('test_budget', None)
        test_budget -= (2 * n)
        self.config['test_budget'] = test_budget
        
        pbar = tqdm.tqdm(total=test_budget)
        num_iterations = 0

        while num_iterations < test_budget:
            input, acc_reward_input, generation = pool.select(self.rng)
            new_generation = generation + 1
            mutant = self.mutate_validate(input, **kwargs)
            acc_reward_mutant, oracle, state_sequence, exec_time = self.mdp(mutant, policy)
            
            sensitivity = None
            if oracle:
                pool.add_crash(mutant)
            elif acc_reward_mutant < acc_reward_input:
                if local_sensitivity:
                    sensitivity = self.local_sensitivity(input, mutant, acc_reward_input, acc_reward_mutant)
                else:
                    sensitivity, _, _, _, _ = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy, generation=new_generation, **kwargs)
                pool.add(mutant, acc_reward_mutant, 0, sensitivity, oracle, generation=new_generation)

            if self.logger is not None:
                self.logger.log(
                    input=mutant,
                    oracle=oracle,
                    reward=acc_reward_mutant,
                    episode_length=len(state_sequence),
                    sensitivity=sensitivity,
                    Generation=new_generation,
                    test_exec_time=exec_time,
                    run_time=time.time()
                )

            num_iterations += 1
            pbar.update(1)

        pbar.close()
        
        if path is not None:
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')

    def random_testing(self, n: int, policy: Any = None, path: str = 'logs', **kwargs):
        '''RT baseline that generates an input at each iteration.'''
        if kwargs.get('exp_name', None) is not None:
            self.config['use_case'] = kwargs['exp_name']
        check_redundant_input = kwargs.get('check_redundant_input', True)

        self.config['name'] = 'RT'
        self.config['test_budget'] = n
        
        if path is not None:
            self.logger = FuzzerLogger(path + '_logs.txt')
            self.logger.write_columns()
        else:
            self.logger = None
            
        pbar = tqdm.tqdm(total=n)
        i = 0
        while i < n:
            execute = True
            random_input = self.sampling(1)

            if check_redundant_input:
                tmp = random_input.tolist()
                if not (tmp in self.evaluated_solutions):
                    self.evaluated_solutions.append(tmp)
                else:
                    execute = False

            if execute:
                acc_reward, oracle, state_sequence, exec_time = self.mdp(random_input, policy)
                if self.logger is not None:
                    self.logger.log(
                        input=random_input,
                        oracle=oracle,
                        reward=acc_reward,
                        episode_length=len(state_sequence),
                        Generation=0,
                        test_exec_time=exec_time,
                        run_time=time.time()
                    )
                pbar.update(1)
                i += 1

        pbar.close()
        if path is not None:
            self.save_configuration(path)

    def save_configuration(self, path: str):
        filepath = path.split('.json')[0]
        self.config['random_state'] = self.rng.bit_generator.state
        with open(filepath + '_config.json', 'w') as f:
            f.write(json.dumps(self.config))

    def save_evaluated_solutions(self, path: str):
        evaluations = np.array(self.evaluated_solutions)
        if np.issubdtype(evaluations.dtype, np.integer):
            np.savetxt(path + '_evaluations.txt', evaluations, fmt='%1.0f', delimiter=',')
        else:
            np.savetxt(path + '_evaluations.txt', evaluations, delimiter=',')

    def load(self, path: str):
        self.coverage_model.load(path)
        config_filepath = path + '_config.json'
        assert os.path.isfile(config_filepath), config_filepath
        with open(config_filepath, 'r') as f:
            config = json.load(f)
        self._load_dict(config)
        self.config = copy.deepcopy(config)
        if os.path.isfile(path + '_evaluations.txt'):
            self.load_evaluated_solutions(path + '_evaluations.txt')
            print('found {} evaluated solutions.'.format(len(self.evaluated_solutions)))

    def _load_dict(self, configuration: Dict):
        self.k = configuration['k']
        self.gamma = configuration['gamma']
        self.random_seed = configuration['random_seed']
        self.env_seed = configuration['env_seed']
        self.rng = np.random.default_rng(self.random_seed)
        self.rng.bit_generator.state = configuration['random_state']

    def load_evaluated_solutions(self, filepath: str):
        self.evaluated_solutions = np.loadtxt(filepath, delimiter=',').tolist()