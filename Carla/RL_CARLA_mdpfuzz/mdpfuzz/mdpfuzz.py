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
        self.rng = np.random.default_rng(self.random_seed) # type: np.random.Generator
        self.coverage_model = CoverageModel(random_seed, k, gamma)
        self.evaluated_solutions = []
        self.generation_map = {} 
        self.executor = executor
        self.sim_steps = self.executor.sim_steps
        self.env_seed = self.executor.env_seed
        self._set_config()

    def _get_key(self, input_arr: np.ndarray) -> str:
        return str(list(input_arr))

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

    def mdp(self, state: np.ndarray, policy: Any = None, generation: int = 0, parent_input: Any = None, phase: str = "Phase1") -> Tuple[float, bool, bool, np.ndarray, float]:
        episode_reward, crash, success, obs_seq, exec_time = self.executor.execute_policy(
            state, policy, generation=generation, parent_input=parent_input, phase=phase
        )
        return episode_reward, crash, success, obs_seq, exec_time

    def sentivity(self, state: np.ndarray, acc_reward: float = None, policy: Any = None, generation: int = 0, parent_input: Any = None, phase: str = "Phase1", **kwargs) -> Tuple[float, float, bool, bool, List[np.ndarray], float]:
        perturbed_state = self.mutate_validate(state, **kwargs)
        perturbation = np.linalg.norm(state - perturbed_state)

        if acc_reward is None:
            acc_reward, crash, success, state_sequence, exec_time = self.mdp(state, policy, generation=generation, parent_input=parent_input, phase=phase)
        else:
            state_sequence = []
            crash = None
            success = None
            exec_time = None

        acc_reward_perturbed, crash_perturbed, success_perturbed, state_sequence_perturbed, exec_time_perturbed = self.mdp(
            perturbed_state, policy, generation=generation, parent_input=state, phase=phase
        )
        
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

        epsilon = 1e-6
        sensitivity = np.abs(acc_reward - acc_reward_perturbed) / (perturbation + epsilon)

        return sensitivity, acc_reward, crash, success, state_sequence, exec_time

    def local_sensitivity(self, state: np.ndarray, state_mutate: np.ndarray, state_reward: float, state_mutate_reward: float):
        perturbation = np.linalg.norm(state - state_mutate)
        epsilon = 1e-6
        return np.abs(state_reward - state_mutate_reward) / (perturbation + epsilon)

    def initialize_coverage_model(self, **kwargs) -> int:
        return 0

    def fuzzing(self, n: int, policy: Any = None, **kwargs):
        # Phase 1: Initialization / Benchmark
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
        
        pbar_init = tqdm.tqdm(total=n, desc="Initializing (Phase 1)")
        model_initialized = False
        total_init_executions = 0
        stop_fuzzing = False # Flag to control interruption

        # [Modified] Phase 1 Loop with try-except for KeyboardInterrupt
        for state in initial_inputs:
            try:
                self.generation_map[self._get_key(state)] = 0
                
                sensitivity, acc_reward, oracle, success, state_sequence, exec_time = self.sentivity(
                    state, policy=policy, generation=0, parent_input=None, phase="Phase1", **kwargs
                )
                
                total_init_executions += 1
                if str(state.tolist()) not in [str(x) for x in self.evaluated_solutions]:
                    self.evaluated_solutions.append(state.tolist())
                
                if not model_initialized:
                    if len(state_sequence) > self.k + 1:
                        self.coverage_model.initialize(state_sequence)
                        model_initialized = True
                        print('[Info] Coverage model initialized with first valid run.')
                    else:
                        print('[Warning] Run too short to initialize coverage model, waiting for next...')

                coverage = 0.0
                if model_initialized:
                    state_sequence_conc = self._concatenate_state_sequence(state_sequence)
                    t0 = time.time()
                    coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
                    coverage_time = time.time() - t0
                else:
                    coverage_time = 0.0

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

                pbar_init.update(1)
            
            except KeyboardInterrupt:
                print("\n[!] User interrupted during Phase 1 (Initialization). Stopping and saving...")
                stop_fuzzing = True
                break

        pbar_init.close()
        
        if not stop_fuzzing:
            print(f"[Info] Initialization finished. Total Init Executions: {total_init_executions}")

            fuzz_start_time = time.time()
            fuzz_iterations = 0

            target_budget_iter = kwargs.get('test_budget', None)
            target_budget_time = kwargs.get('test_budget_in_seconds', None)
            
            if target_budget_time is not None:
                self.config['test_budget_in_seconds'] = target_budget_time
                pbar = tqdm.tqdm(total=target_budget_time, unit='s', desc="Fuzzing (Time)")
            else:
                self.config['test_budget'] = target_budget_iter
                pbar = tqdm.tqdm(total=target_budget_iter, desc="Fuzzing (Iter)")
                
            try:
                while True:
                    if target_budget_time:
                        current_fuzz_duration = time.time() - fuzz_start_time
                        if current_fuzz_duration > target_budget_time:
                            print(f"[Info] Time budget reached: {current_fuzz_duration:.2f}s > {target_budget_time}s")
                            break
                    elif target_budget_iter:
                        if fuzz_iterations >= target_budget_iter:
                            print(f"[Info] Iteration budget reached: {fuzz_iterations} >= {target_budget_iter}")
                            break

                    if len(pool.inputs) == 0:
                        print("[Error] Pool is empty, cannot fuzz.")
                        break

                    input, acc_reward_input = pool.select(self.rng)
                    
                    parent_key = self._get_key(input)
                    parent_gen = self.generation_map.get(parent_key, 0)
                    current_gen = parent_gen + 1
                    
                    mutant = self.mutate_validate(input, **kwargs)
                    
                    acc_reward_mutant, oracle, success, state_sequence, exec_time = self.mdp(
                        mutant, policy, generation=current_gen, parent_input=input, phase="Phase2"
                    )
                    
                    fuzz_iterations += 1
                    
                    coverage = 0.0
                    if model_initialized:
                        state_sequence_conc = self._concatenate_state_sequence(state_sequence)
                        t0 = time.time()
                        coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
                        coverage_time = time.time() - t0
                    else:
                        if len(state_sequence) > self.k + 1:
                            self.coverage_model.initialize(state_sequence)
                            model_initialized = True
                            print('[Info] Coverage model initialized during Fuzzing phase.')
                        coverage_time = 0.0

                    sensitivity = None
                    if oracle:
                        pool.add_crash(mutant)
                        self.generation_map[self._get_key(mutant)] = current_gen
                    elif (acc_reward_mutant < acc_reward_input) or (coverage < self.tau):
                        if local_sensitivity:
                            sensitivity = self.local_sensitivity(input, mutant, acc_reward_input, acc_reward_mutant)
                        else:
                            sensitivity, _acc_reward_mutant_copy, _none_oracle, _success_flag, _empty_list, _none_exec_time = self.sentivity(
                                mutant, acc_reward=acc_reward_mutant, policy=policy, generation=current_gen, parent_input=input, phase="Phase2", **kwargs
                            )
                        
                        pool.add(mutant, acc_reward_mutant, coverage, sensitivity, oracle)
                        self.generation_map[self._get_key(mutant)] = current_gen

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

                    if target_budget_time:
                        current_elapsed_int = int(time.time() - fuzz_start_time)
                        increment = current_elapsed_int - pbar.n
                        if increment > 0:
                            pbar.update(increment)
                    else:
                        pbar.update(1)
            
            # [Modified] Catch KeyboardInterrupt specifically to allow graceful exit
            except KeyboardInterrupt:
                print("\n[!] User interrupted during Phase 2 (Fuzzing). Saving results and exiting...")
            
            except Exception as e:
                print(f"[Error in Fuzzing Loop] {e}")
                import traceback
                traceback.print_exc()

            pbar.close()

        if path is not None:
            print("[Info] Saving final configuration and results...")
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')
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

        initial_inputs = self.sampling(n).tolist()
        self.config['init_budget'] = n
        if kwargs.get('light_pool', False):
            pool = LightPool() 
        else:
            pool = IndexedPool(is_integer=np.issubdtype(np.array(initial_inputs).dtype, np.integer)) 
        
        pbar_init = tqdm.tqdm(total=n, desc="Initializing (Strict Phase 1)")
        
        successful_seeds = 0
        input_idx = 0
        total_init_executions = 0
        stop_fuzzing = False
        
        # [Modified] Phase 1 Loop with try-except for KeyboardInterrupt
        while successful_seeds < n:
            try:
                if input_idx >= len(initial_inputs):
                    initial_inputs.append(self.sampling(1).tolist())
                
                state = np.array(initial_inputs[input_idx])
                input_idx += 1
                
                acc_reward, oracle, is_success, state_sequence, exec_time = self.mdp(
                    state, policy, generation=0, parent_input=None, phase="Phase1"
                )
                total_init_executions += 1 

                if str(state.tolist()) not in [str(x) for x in self.evaluated_solutions]:
                    self.evaluated_solutions.append(state.tolist())
                
                if is_success:
                    sensitivity, _, _, _, _, _ = self.sentivity(
                        state, 
                        acc_reward=acc_reward, 
                        policy=policy, 
                        generation=0, 
                        parent_input=None, 
                        phase="Phase1",
                        **kwargs
                    )
                    
                    if sensitivity == 0:
                        sensitivity = 1e-6
                    
                    pool.add(state, acc_reward, 0, sensitivity, oracle)
                    self.generation_map[self._get_key(state)] = 0
                    successful_seeds += 1
                    pbar_init.update(1)
                
                if self.logger is not None:
                    episode_length = len(state_sequence)
                    self.logger.log(
                        input=state,
                        oracle=oracle,
                        reward=acc_reward,
                        episode_length=episode_length,
                        sensitivity=0.0, 
                        test_exec_time=exec_time,
                        run_time=time.time()
                    )
            except KeyboardInterrupt:
                print("\n[!] User interrupted during Phase 1 (Strict). Stopping and saving...")
                stop_fuzzing = True
                break

        pbar_init.close()

        if not stop_fuzzing:
            print(f"[Info] Initialization finished. Total Init Executions: {total_init_executions} (Target Success: {n})")
            print("[Info] Note: Initialization cost is NOT deducted from Fuzzing budget.")

            fuzz_start_time = time.time()
            fuzz_iterations = 0

            target_budget_iter = kwargs.get('test_budget', None)
            target_budget_time = kwargs.get('test_budget_in_seconds', None)
            
            if target_budget_time is not None:
                self.config['test_budget_in_seconds'] = target_budget_time
                pbar = tqdm.tqdm(total=target_budget_time, unit='s', desc="Fuzzing (Time)")
            else:
                self.config['test_budget'] = target_budget_iter
                pbar = tqdm.tqdm(total=target_budget_iter, desc="Fuzzing (Iter)")

            # [Modified] Phase 2 Loop with try-except for KeyboardInterrupt
            try:
                while True:
                    if target_budget_time:
                        current_fuzz_duration = time.time() - fuzz_start_time
                        if current_fuzz_duration > target_budget_time:
                            print(f"[Info] Time budget reached: {current_fuzz_duration:.2f}s > {target_budget_time}s")
                            break
                    elif target_budget_iter:
                        if fuzz_iterations >= target_budget_iter:
                            print(f"[Info] Iteration budget reached: {fuzz_iterations} >= {target_budget_iter}")
                            break

                    if len(pool.inputs) == 0:
                        print("[Error] Pool is empty, cannot fuzz.")
                        break

                    input, acc_reward_input = pool.select(self.rng)
                    
                    parent_key = self._get_key(input)
                    parent_gen = self.generation_map.get(parent_key, 0)
                    current_gen = parent_gen + 1
                    
                    mutant = self.mutate_validate(input, **kwargs)
                    acc_reward_mutant, oracle, success, state_sequence, exec_time = self.mdp(
                        mutant, policy, generation=current_gen, parent_input=input, phase="Phase2"
                    )
                    
                    fuzz_iterations += 1
                    
                    sensitivity = None
                    if oracle:
                        pool.add_crash(mutant)
                        self.generation_map[self._get_key(mutant)] = current_gen
                    elif acc_reward_mutant < acc_reward_input:
                        if local_sensitivity:
                            sensitivity = self.local_sensitivity(input, mutant, acc_reward_input, acc_reward_mutant)
                        else:
                            sensitivity, _acc_reward_mutant_copy, _none_oracle, _success_flag, _empty_list, _none_exec_time = self.sentivity(
                                mutant, acc_reward=acc_reward_mutant, policy=policy, generation=current_gen, parent_input=input, phase="Phase2", **kwargs
                            )
                        
                        pool.add(mutant, acc_reward_mutant, 0, sensitivity, oracle)
                        self.generation_map[self._get_key(mutant)] = current_gen

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

                    if target_budget_time:
                        current_elapsed_int = int(time.time() - fuzz_start_time)
                        increment = current_elapsed_int - pbar.n
                        if increment > 0:
                            pbar.update(increment)
                    else:
                        pbar.update(1)
            
            except KeyboardInterrupt:
                print("\n[!] User interrupted during Phase 2 (Fuzzing). Saving results and exiting...")
            
            pbar.close()

        if path is not None:
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')
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
        if len(evaluations) > 0 and np.issubdtype(evaluations.dtype, np.integer):
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

    def load_evaluated_solutions(self, filepath: str):
        self.evaluated_solutions = np.loadtxt(filepath, delimiter=',').tolist()

    def random_testing(self, n: int, policy: Any = None, path: str = 'logs', **kwargs):
        if kwargs.get('exp_name', None) is not None:
            self.config['use_case'] = kwargs['exp_name']
        check_redundant_input = kwargs.get('check_redundant_input', True)

        self.config['name'] = 'RT'
        
        target_budget_time = kwargs.get('test_budget_in_seconds', None)
        
        if target_budget_time is not None:
            self.config['test_budget_in_seconds'] = target_budget_time
            self.config['test_budget'] = None # Explicitly set to None to avoid confusion
            pbar = tqdm.tqdm(total=target_budget_time, unit='s', desc="Random Testing (Time)")
        else:
            self.config['test_budget'] = n
            pbar = tqdm.tqdm(total=n, desc="Random Testing (Iter)")
            
        self.logger = FuzzerLogger(path + '_logs.txt')
        self.logger.write_columns()
        
        start_time = time.time()
        i = 0
        
        # [Modified] Loop with try-except for KeyboardInterrupt and heartbeat logs
        try:
            while True:
                if target_budget_time:
                    current_duration = time.time() - start_time
                    if current_duration > target_budget_time:
                        print(f"[Info] Time budget reached: {current_duration:.2f}s > {target_budget_time}s")
                        break
                elif i >= n:
                    break
                
                # [Fix] Added debug logging to confirm liveness
                # print(f"[Debug] Iteration {i}: Generating input...", end='\r')

                execute = True
                random_input = self.sampling(1)

                if check_redundant_input:
                    tmp = random_input.tolist()
                    if not (tmp in self.evaluated_solutions):
                        self.evaluated_solutions.append(tmp)
                    else:
                        execute = False

                if execute:
                    # [Fix] Added debug logging to confirm liveness
                    # print(f"[Debug] Iteration {i}: Executing policy...", end='\r')

                    acc_reward, oracle, success, state_sequence, exec_time = self.mdp(random_input, policy, phase="RT")

                    coverage = 0.0
                    coverage_time = 0.0

                    episode_length = len(state_sequence)
                    self.logger.log(
                        input=random_input,
                        oracle=oracle,
                        reward=acc_reward,
                        episode_length=episode_length,
                        test_exec_time=exec_time,
                        coverage=coverage, 
                        coverage_time=coverage_time,
                        run_time=time.time()
                    )
                    
                    i += 1
                    
                    # [Fix] Periodic heartbeat log to confirm process is not frozen
                    if i % 100 == 0:
                        print(f"[Info] Random Testing alive. Completed {i} iterations.")
                
                if target_budget_time:
                    current_elapsed_int = int(time.time() - start_time)
                    increment = current_elapsed_int - pbar.n
                    if increment > 0:
                        pbar.update(increment)
                else:
                    if execute:
                        pbar.update(1)
        
        except KeyboardInterrupt:
             print("\n[!] User interrupted during Random Testing. Saving results and exiting...")

        pbar.close()
        self.save_configuration(path)
        self.save_evaluated_solutions(path)