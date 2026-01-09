import os
import time
import copy
import json
import tqdm
import numpy as np

from typing import List, Tuple, Dict, Any

# or runs python -m mdpfuzz.py
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
        self.mutation_history = [] 

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


    def mdp(self, state: np.ndarray, policy: Any = None) -> Tuple[float, bool, np.ndarray, float, float, float]:
        '''Returns the accumulated reward, crash bool, state sequence, exec time, and BD metrics.'''
        # [修改] 解包 6 个返回值
        episode_reward, done, obs_seq, exec_time, bd_dist, bd_angle = self.executor.execute_policy(state, policy)
        return episode_reward, done, obs_seq, exec_time, bd_dist, bd_angle


    def sentivity(self, state: np.ndarray, acc_reward: float = None, policy: Any = None, generation: int = None ,**kwargs) -> Tuple[float, float, bool, List[np.ndarray], float, float, float]:
        '''
        Computes the sensitivity of the state @state.
        '''
        perturbed_state = self.mutate_validate(state, **kwargs)
        perturbation = np.linalg.norm(state - perturbed_state)

        bd_dist_ret = None
        bd_angle_ret = None

        if acc_reward is None:
            # [修改] 解包
            acc_reward, crash, state_sequence, exec_time, bd_dist_ret, bd_angle_ret = self.mdp(state, policy)
        else:
            state_sequence = []
            crash = None
            exec_time = None
            bd_dist_ret = None
            bd_angle_ret = None

        # [修改] 解包 perturbed
        acc_reward_perturbed, crash_perturbed, state_sequence_perturbed, exec_time_perturbed, bd_dist_p, bd_angle_p = self.mdp(perturbed_state, policy)
        
        if self.logger is not None:
            episode_length = len(state_sequence_perturbed)
            self.logger.log(
                input=perturbed_state,
                oracle=crash_perturbed,
                reward=acc_reward_perturbed,
                episode_length=episode_length,
                Generation=generation,
                test_exec_time=exec_time_perturbed,
                run_time=time.time(),
                # [新增] 记录 BD 指标
                bd_distance=bd_dist_p,
                bd_mean_angle=bd_angle_p
            )

        sensitivity = np.abs(acc_reward - acc_reward_perturbed) / perturbation

        # [修改] 返回值包含 BD 指标
        return sensitivity, acc_reward, crash, state_sequence, exec_time, bd_dist_ret, bd_angle_ret


    def local_sensitivity(self, state: np.ndarray, state_mutate: np.ndarray, state_reward: float, state_mutate_reward: float):
        perturbation = np.linalg.norm(state - state_mutate)
        return np.abs(state_reward - state_mutate_reward) / perturbation


    def initialize_coverage_model(self, **kwargs) -> int:
        exec_counter = kwargs.get('exec_counter', 0)
        state_sequence = kwargs.pop('state_sequence', None)
        if state_sequence is None:
            policy = kwargs.get('policy', None)
            random_input = kwargs.get('input', self.sampling())
            # [修改] 解包
            reward, crash, state_sequence, exec_time, bd_dist, bd_angle = self.mdp(random_input, policy)
            exec_counter += 1
            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=random_input,
                    oracle=crash,
                    reward=reward,
                    episode_length=episode_length,
                    Generation=0,
                    test_exec_time=exec_time,
                    run_time=time.time(),
                    # [新增]
                    bd_distance=bd_dist,
                    bd_mean_angle=bd_angle
                    )

        if len(state_sequence) < self.k + 1:
            kwargs['exec_counter'] = exec_counter
            return self.initialize_coverage_model(**kwargs)
        else:
            self.coverage_model.initialize(state_sequence)
        print('Coverage model initialized')
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
        if kwargs.get('light_pool', False):
            pool = LightPool()
        else:
            pool = IndexedPool(is_integer=np.issubdtype(initial_inputs.dtype, np.integer))

        num_initial_executions = self.initialize_coverage_model(policy=policy)
        self.config['num_initial_executions'] = num_initial_executions
        pbar = tqdm.tqdm(total=n)
        
        for state in initial_inputs:
            # [修改] 解包 sentivity 返回值
            sensitivity, acc_reward, oracle, state_sequence, exec_time, bd_dist, bd_angle = self.sentivity(state, policy=policy, generation=0, **kwargs)
            state_sequence_conc = self._concatenate_state_sequence(state_sequence)
            t0 = time.time()
            coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
            coverage_time = time.time() - t0
            pool.add(state, acc_reward, coverage, sensitivity, oracle, generation=0)

            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=state,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=episode_length,
                    sensitivity=sensitivity,
                    coverage=coverage,
                    Generation=0,
                    test_exec_time=exec_time,
                    coverage_time=coverage_time,
                    run_time=time.time(),
                    # [新增]
                    bd_distance=bd_dist,
                    bd_mean_angle=bd_angle
                )

            if oracle:
                pool.add_crash(state)

            pbar.update(1)
        pbar.close()

        # Fuzzing Loop
        test_budget_in_seconds = kwargs.get('test_budget_in_seconds', None)
        if test_budget_in_seconds is None:
            test_budget = kwargs.get('test_budget', None)
            assert test_budget is not None
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
                    if num_iterations == test_budget: break
                else:
                    if (current_time - start_time) > test_budget_in_seconds: break

                input, acc_reward_input, generation = pool.select(self.rng)
                new_generation = generation + 1
                mutant = self.mutate_validate(input, **kwargs)
                
                # [修改] 解包 mdp
                acc_reward_mutant, oracle, state_sequence, exec_time, bd_dist, bd_angle = self.mdp(mutant, policy)

                record = np.concatenate([input, mutant, np.array([int(oracle)])])
                self.mutation_history.append(record)

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
                        # [修改] 解包 sentivity
                        sensitivity, _, _, _, _, _, _ = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy, generation=new_generation, **kwargs)
                    pool.add(mutant, acc_reward_mutant, coverage, sensitivity, oracle, generation=new_generation)

                if self.logger is not None:
                    episode_length = len(state_sequence)
                    self.logger.log(
                        input=mutant,
                        oracle=oracle,
                        reward=acc_reward_mutant,
                        episode_length=episode_length,
                        sensitivity=sensitivity,
                        coverage=coverage,
                        Generation=new_generation,
                        test_exec_time=exec_time,
                        coverage_time=coverage_time,
                        run_time=time.time(),
                        # [新增]
                        bd_distance=bd_dist,
                        bd_mean_angle=bd_angle
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
            import traceback
            traceback.print_exc()

        pbar.close()
        
        if path is not None:
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')
            self.save_mutation_history(path)
            if not kwargs.get('save_logs_only', False):
                self.coverage_model.save(path)
                self.save_evaluated_solutions(path)
                if not kwargs.get('light_pool', False):
                    pool.save(path)


    def fuzzing_no_coverage(self, n: int, policy: Any = None, **kwargs):
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
            pool = LightPool()
        else:
            pool = IndexedPool(is_integer=np.issubdtype(initial_inputs.dtype, np.integer))

        pbar = tqdm.tqdm(total=n)
        for state in initial_inputs:
            # [修改] 解包
            sensitivity, acc_reward, oracle, state_sequence, exec_time, bd_dist, bd_angle = self.sentivity(state, policy=policy, generation=0, **kwargs)
            pool.add(state, acc_reward, 0, sensitivity, oracle, generation=0)

            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=state,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=episode_length,
                    sensitivity=sensitivity,
                    Generation=0,
                    test_exec_time=exec_time,
                    run_time=time.time(),
                    # [新增]
                    bd_distance=bd_dist,
                    bd_mean_angle=bd_angle
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
                if num_iterations == test_budget: break
            else:
                if (current_time - start_time) > test_budget_in_seconds: break

            input, acc_reward_input, generation = pool.select(self.rng)
            new_generation = generation + 1
            mutant = self.mutate_validate(input, **kwargs)
            
            # [修改] 解包 mdp
            acc_reward_mutant, oracle, state_sequence, exec_time, bd_dist, bd_angle = self.mdp(mutant, policy)
            
            record = np.concatenate([input, mutant, np.array([int(oracle)])])
            self.mutation_history.append(record)
            
            sensitivity = None
            if oracle:
                pool.add_crash(mutant)
            elif acc_reward_mutant < acc_reward_input:
                if local_sensitivity:
                    sensitivity = self.local_sensitivity(input, mutant, acc_reward_input, acc_reward_mutant)
                else:
                    # [修改] 解包
                    sensitivity, _, _, _, _, _, _ = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy, generation=new_generation, **kwargs)
                pool.add(mutant, acc_reward_mutant, 0, sensitivity, oracle, generation=new_generation)

            if self.logger is not None:
                episode_length = len(state_sequence)
                self.logger.log(
                    input=mutant,
                    oracle=oracle,
                    reward=acc_reward_mutant,
                    episode_length=episode_length,
                    sensitivity=sensitivity,
                    Generation=new_generation,
                    test_exec_time=exec_time,
                    run_time=time.time(),
                    # [新增]
                    bd_distance=bd_dist,
                    bd_mean_angle=bd_angle
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
            self.save_mutation_history(path)
            if not kwargs.get('save_logs_only', False):
                self.save_evaluated_solutions(path)
                if not kwargs.get('light_pool', False):
                    pool.save(path)


    def save_configuration(self, path: str):
         filepath = path.split('.json')[0]
         self.config['random_state'] = self.rng.bit_generator.state
         with open(filepath + '_config.json', 'w') as f:
             f.write(json.dumps(self.config))


    def save_evaluated_solutions(self, path: str):
         evaluations = np.array(self.evaluated_solutions)
         if len(evaluations) > 0:
             if np.issubdtype(evaluations.dtype, np.integer):
                 np.savetxt(path + '_evaluations.txt', evaluations, fmt='%1.0f', delimiter=',')
             else:
                 np.savetxt(path + '_evaluations.txt', evaluations, delimiter=',')

    def save_mutation_history(self, path: str):
         if len(self.mutation_history) > 0:
            print(f"Saving mutation history ({len(self.mutation_history)} entries) to {path}_mutations.txt")
            with open(path + '_mutations.txt', 'w') as f:
                f.write("ParentState; MutantState; Oracle\n")
                for record in self.mutation_history:
                    state_dim = (len(record) - 1) // 2
                    parent = record[:state_dim]
                    mutant = record[state_dim : 2*state_dim]
                    oracle = int(record[-1])
                    parent_str = np.array2string(parent, separator=',').replace('\n', '')
                    mutant_str = np.array2string(mutant, separator=',').replace('\n', '')
                    f.write(f"{parent_str}; {mutant_str}; {oracle}\n")
         else:
            print("No mutation history to save.")


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


    def random_testing(self, n: int, policy: Any = None, path: str = 'logs', **kwargs):
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
                # [修改] 解包
                acc_reward, oracle, state_sequence, exec_time, bd_dist, bd_angle = self.mdp(random_input, policy)
                episode_length = len(state_sequence)
                self.logger.log(
                    input=random_input,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=episode_length,
                    Generation=0,
                    test_exec_time=exec_time,
                    run_time=time.time(),
                    # [新增]
                    bd_distance=bd_dist,
                    bd_mean_angle=bd_angle
                )
                pbar.update(1)
                i += 1
        pbar.close()
        self.save_configuration(path)
        self.save_evaluated_solutions(path)