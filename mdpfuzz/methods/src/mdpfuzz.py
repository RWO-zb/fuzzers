import os
import time
import copy
import json
import tqdm
import numpy as np

from typing import List, Tuple, Dict, Any

# or runs python -m mdpfuzz.py
if __package__ is None or __package__ == '':
    # uses current directory visibility
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
        # in order, k is the number of components (or clusters) for the 2 GMMs, tau is the density threshold to update the GMMs and gamma is the weight of the update
        self.k = k
        self.tau = tau
        self.gamma = gamma

        # random generators
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed) # type: np.random.Generator

        # coverage model (composed of 2 GMMS)
        self.coverage_model = CoverageModel(random_seed, k, gamma)
        # used to track uniqueness of solutions
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
        '''Returns a list of @n inputs randomly generated.'''
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
        '''Returns the accumulated reward, whether a crash is detected and the state sequence.'''
        episode_reward, done, obs_seq, exec_time = self.executor.execute_policy(state, policy)
        return episode_reward, done, obs_seq, exec_time


    def sentivity(self, state: np.ndarray, acc_reward: float = None, policy: Any = None, **kwargs) -> Tuple[float, float, bool, List[np.ndarray], float]:
        '''
        Computes the sensitivity of the state @state.
        It first perturbs the state and computes the perturbation quantity.
        Then, the two states are executed and the sensitivity is computed.
        It returns the latter, as well as the results of the execution for the state (acc. reward, sequence, oracle and execution time).
        '''
        # perturbs the state and computes the perturbation
        perturbed_state = self.mutate_validate(state, **kwargs)
        perturbation = np.linalg.norm(state - perturbed_state)

        # runs the two states if no accumulated reward is provided
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
                test_exec_time=exec_time_perturbed,
                run_time=time.time()
            )

        # computes the sensitivity, the coverage, and adds test case in the pool
        sensitivity = np.abs(acc_reward - acc_reward_perturbed) / perturbation

        return sensitivity, acc_reward, crash, state_sequence, exec_time


    def local_sensitivity(self, state: np.ndarray, state_mutate: np.ndarray, state_reward: float, state_mutate_reward: float):
        perturbation = np.linalg.norm(state - state_mutate)
        return np.abs(state_reward - state_mutate_reward) / perturbation


    def initialize_coverage_model(self, **kwargs) -> int:
        '''Initializes the coverage model and returns the number of executions that have been done.'''
        exec_counter = kwargs.get('exec_counter', 0)
        state_sequence = kwargs.pop('state_sequence', None)
        if state_sequence is None:
            policy = kwargs.get('policy', None)
            random_input = kwargs.get('input', self.sampling())
            reward, crash, state_sequence, exec_time = self.mdp(random_input, policy)
            exec_counter += 1
            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=random_input,
                    oracle=crash,
                    reward=reward,
                    episode_length=episode_length,
                    test_exec_time=exec_time,
                    run_time=time.time()
                    )

        # it needs at least k + 1 states (for gmm_c)
        if len(state_sequence) < self.k + 1:
            kwargs['exec_counter'] = exec_counter
            return self.initialize_coverage_model(**kwargs)
        else:
            self.coverage_model.initialize(state_sequence)
        print('Coverage model initialized')
        return exec_counter


    def fuzzing(self, n: int, policy: Any = None, **kwargs):
        '''
        Conducts fuzzing to generate test cases for the system under test (SUT).

        Args:
        - n (int): Number of iterations for fuzzing.
        - policy (tt.TestAgent): The testing policy or agent guiding the fuzzing process.
        - saving_path (str, optional): Path to save logs and results (default: None).
        - local_sensitivity (bool, optional): Flag indicating whether to compute local sensitivity (default: False).
        - test_budget_in_seconds (int, optional): Time budget for fuzzing in seconds (default: None).
        - test_budget (int, optional): Number of iterations if time budget is not specified (default: None).
        - exp_name (str, optional): Name of the experiment to overwrite the key "use_case" of the configuration dictionary.
        - save_logs_only (bool, optional): Flag indicating whether to only save the configuration of the execution (default: False).
        - light_pool (bool, optional): Flag indicating whether to use the LightPool class for the pool (default: False).

        Returns:
        None. The function conducts the fuzzing process and stores generated test cases.
        '''
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
        if kwargs.get('light_pool', False):
            pool = LightPool() # type: Pool
        else:
            pool = IndexedPool(is_integer=np.issubdtype(initial_inputs.dtype, np.integer)) # type: Pool

        # initializes the coverage model by running the policy on a randomly generated input to sample states of the MDP
        num_initial_executions = self.initialize_coverage_model(policy=policy)
        self.config['num_initial_executions'] = num_initial_executions
        pbar = tqdm.tqdm(total=n)
        for state in initial_inputs:
            sensitivity, acc_reward, oracle, state_sequence, exec_time = self.sentivity(state, policy=policy, **kwargs)
            state_sequence_conc = self._concatenate_state_sequence(state_sequence)
            # computes the coverage and adds test case in the pool
            t0 = time.time()
            coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
            coverage_time = time.time() - t0
            pool.add(state, acc_reward, coverage, sensitivity, oracle)

            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=state,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=episode_length,
                    sensitivity=sensitivity,
                    coverage=coverage,
                    test_exec_time=exec_time,
                    coverage_time=coverage_time,
                    run_time=time.time()
                )

            if oracle:
                pool.add_crash(state)

            pbar.update(1)
        pbar.close()

        test_budget_in_seconds = kwargs.get('test_budget_in_seconds', None)
        if test_budget_in_seconds is None:
            test_budget = kwargs.get('test_budget', None)
            assert test_budget is not None
            # accounts for the cost of the initialization
            test_budget -=  (2 * n) + num_initial_executions
            pbar = tqdm.tqdm(total=test_budget)
            self.config['test_budget'] = test_budget
            num_iterations = 0
        else:
            start_time = time.time()
            current_time = time.time()
            seconds = 0
            pbar = tqdm.tqdm(total=test_budget_in_seconds)
            self.config['test_budget_in_seconds'] = test_budget_in_seconds
        try:
            while True:
                if test_budget_in_seconds is None:
                    if num_iterations == test_budget:
                        break
                else:
                    if (current_time - start_time) > test_budget_in_seconds:
                        break

                input, acc_reward_input = pool.select(self.rng)
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
                        sensitivity, _acc_reward_mutant_copy, _none_oracle, _empty_list, _none_exec_time = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy, **kwargs)
                    pool.add(mutant, acc_reward_mutant, coverage, sensitivity, oracle)

                if self.logger is not None:
                    episode_length = len(state_sequence)
                    self.logger.log(
                        input=mutant,
                        oracle=oracle,
                        reward=acc_reward_mutant,
                        episode_length=episode_length,
                        sensitivity=sensitivity,
                        coverage=coverage,
                        test_exec_time=exec_time,
                        coverage_time=coverage_time,
                        run_time=time.time()
                    )

                if test_budget_in_seconds is None:
                    num_iterations += 1
                    pbar.update(1)
                else:
                    current_time = time.time()
                    if int(current_time - start_time) > seconds:
                        seconds += 1
                        pbar.update(1)
        except Exception as e:
            print(e)

        pbar.close()
        # saves at least the configuration and the history of the input selection (if Pool allows)
        if path is not None:
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')
            if not kwargs.get('save_logs_only', False):
                self.coverage_model.save(path)
                self.save_evaluated_solutions(path)
                # saves pool only if not LightPool
                if not kwargs.get('light_pool', False):
                    pool.save(path)


    def fuzzing_no_coverage(self, n: int, policy: Any = None, **kwargs):
        '''
        Works similarly as fuzzing but coverages are not computed.
        '''
        if kwargs.get('exp_name', None) is not None:
            self.config['use_case'] = kwargs['exp_name']
        self.config['name'] = 'Fuzzer'
        path = kwargs.get('saving_path', None)
        if path is not None:
            self.logger = FuzzerLogger(path + '_logs.txt')
            self.logger.write_columns()
        else:
            self.logger = None

        local_sensitivity = kwargs.get('local_sensitivity', False)

        initial_inputs = self.sampling(n)
        self.config['init_budget'] = n
        if kwargs.get('light_pool', False):
            pool = LightPool() # type: Pool
        else:
            pool = IndexedPool(is_integer=np.issubdtype(initial_inputs.dtype, np.integer)) # type: Pool
        pbar = tqdm.tqdm(total=n)
        for state in initial_inputs:
            sensitivity, acc_reward, oracle, state_sequence, exec_time = self.sentivity(state, policy=policy, **kwargs)
            pool.add(state, acc_reward, 0, sensitivity, oracle)

            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=state,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=episode_length,
                    sensitivity=sensitivity,
                    test_exec_time=exec_time,
                    run_time=time.time()
                )

            if oracle:
                pool.add_crash(state)

            pbar.update(1)
        pbar.close()

        test_budget_in_seconds = kwargs.get('test_budget_in_seconds', None)
        if test_budget_in_seconds is None:
            test_budget = kwargs.get('test_budget', None)
            assert test_budget is not None
            test_budget -=  (2 * n)
            pbar = tqdm.tqdm(total=test_budget)
            self.config['test_budget'] = test_budget
            num_iterations = 0
        else:
            start_time = time.time()
            current_time = time.time()
            seconds = 0
            pbar = tqdm.tqdm(total=test_budget_in_seconds)
            self.config['test_budget_in_seconds'] = test_budget_in_seconds

        while True:
            if test_budget_in_seconds is None:
                if num_iterations == test_budget:
                    break
            else:
                if (current_time - start_time) > test_budget_in_seconds:
                    break

            input, acc_reward_input = pool.select(self.rng)
            mutant = self.mutate_validate(input, **kwargs)
            acc_reward_mutant, oracle, state_sequence, exec_time = self.mdp(mutant, policy)
            sensitivity = None
            if oracle:
                pool.add_crash(mutant)
            elif acc_reward_mutant < acc_reward_input:
                if local_sensitivity:
                    sensitivity = self.local_sensitivity(input, mutant, acc_reward_input, acc_reward_mutant)
                else:
                    sensitivity, _acc_reward_mutant_copy, _none_oracle, _empty_list, _none_exec_time = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy, **kwargs)
                pool.add(mutant, acc_reward_mutant, 0, sensitivity, oracle)

            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=mutant,
                    oracle=oracle,
                    reward=acc_reward_mutant,
                    episode_length=episode_length,
                    sensitivity=sensitivity,
                    test_exec_time=exec_time,
                    run_time=time.time()
                )

            if test_budget_in_seconds is None:
                num_iterations += 1
                pbar.update(1)
            else:
                current_time = time.time()
                if int(current_time - start_time) > seconds:
                    seconds += 1
                    pbar.update(1)

        pbar.close()
        if path is not None:
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')
            if not kwargs.get('save_logs_only', False):
                self.save_evaluated_solutions(path)
                # saves pool only if not LightPool
                if not kwargs.get('light_pool', False):
                    pool.save(path)


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
        self.rng = np.random.default_rng(self.random_seed) # type: np.random.Generator
        self.rng.bit_generator.state = configuration['random_state']
        # self._set_config()


    def load_evaluated_solutions(self, filepath: str):
        self.evaluated_solutions = np.loadtxt(filepath, delimiter=',').tolist()


    def random_testing(self, n: int, policy: Any = None, path: str = 'logs', **kwargs):
        '''
        RT baseline that generates an input at each iteration.
        By default, the method checks at the inputs don't have been tested before.
        Such redundant testing guard can be disabled with the argument 'check_redundant_input'.
        '''
        if kwargs.get('exp_name', None) is not None:
            self.config['use_case'] = kwargs['exp_name']
        check_redundant_input = kwargs.get('check_redundant_input', True)

        self.config['name'] = 'RT'
        self.config['test_budget'] = n
        self.logger = FuzzerLogger(path + '_logs.txt')
        self.logger.write_columns()
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
                episode_length = len(state_sequence)
                self.logger.log(
                    input=random_input,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=episode_length,
                    test_exec_time=exec_time,
                    run_time=time.time()
                )
                pbar.update(1)
                i += 1

        pbar.close()

        self.save_configuration(path)
        self.save_evaluated_solutions(path)