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
                run_time=time.time(),
                seed_id=kwargs.get('seed_id', None) # 传递 seed_id
            )

        sensitivity = np.abs(acc_reward - acc_reward_perturbed) / perturbation
        return sensitivity, acc_reward, crash, state_sequence, exec_time

    def local_sensitivity(self, state: np.ndarray, state_mutate: np.ndarray, state_reward: float, state_mutate_reward: float):
        perturbation = np.linalg.norm(state - state_mutate)
        return np.abs(state_reward - state_mutate_reward) / perturbation

    # [修改] 增加 seed_id 参数并写入 JSON
    def _save_observations(self, path: str, input_data: np.ndarray, oracle: bool, obs_seq: np.ndarray, generation: int, seed_id: int = None):
        if path is None:
            return
        file_path = path + '_obs.txt'
        with open(file_path, 'a') as f:
            header_info = {
                "Generation": generation,
                "Input": input_data.tolist() if isinstance(input_data, np.ndarray) else input_data,
                "Oracle": bool(oracle),
                "Steps": len(obs_seq),
                "SeedID": seed_id # [新增] 记录 SeedID 以便后续绘图映射
            }
            f.write(f"--- Test Case Info: {json.dumps(header_info)} ---\n")
            np.savetxt(f, obs_seq, fmt='%.6f', delimiter=', ')
            f.write("\n")

    def initialize_coverage_model(self, **kwargs) -> int:
        exec_counter = kwargs.get('exec_counter', 0)
        state_sequence = kwargs.pop('state_sequence', None)
        path = kwargs.get('saving_path', None) 

        if state_sequence is None:
            policy = kwargs.get('policy', None)
            random_input = kwargs.get('input', self.sampling())
            reward, crash, state_sequence, exec_time = self.mdp(random_input, policy)
            exec_counter += 1
            # 初始化阶段通常没有 seed_id，或者可以视作它们自己就是种子
            self._save_observations(path, random_input, crash, state_sequence, 0, seed_id=None)

            if self.logger is not None:
                self.logger.log(
                    input=random_input,
                    oracle=crash,
                    reward=reward,
                    episode_length=len(state_sequence),
                    Generation=0,
                    test_exec_time=exec_time,
                    run_time=time.time(),
                    seed_id=None 
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

        kwargs['saving_path'] = path
        num_initial_executions = self.initialize_coverage_model(policy=policy, **kwargs)
        self.config['num_initial_executions'] = num_initial_executions
        
        pbar = tqdm.tqdm(total=n)
        for i, state in enumerate(initial_inputs): # 使用 enumerate 获取 seed_id
            sensitivity, acc_reward, oracle, state_sequence, exec_time = self.sentivity(state, policy=policy, generation=0, seed_id=i, **kwargs)
            # [修改] 传入 seed_id
            self._save_observations(path, state, oracle, state_sequence, 0, seed_id=i)

            state_sequence_conc = self._concatenate_state_sequence(state_sequence)
            t0 = time.time()
            coverage = self.coverage_model.sequence_freshness(state_sequence, state_sequence_conc, tau=self.tau)
            coverage_time = time.time() - t0
            pool.add(state, acc_reward, coverage, sensitivity, oracle, generation=0, seed_id=i) # 传入 seed_id

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
                    run_time=time.time(),
                    seed_id=i # 传入 seed_id
                )
            if oracle:
                pool.add_crash(state)
            pbar.update(1)
        pbar.close()

        test_budget = kwargs.get('test_budget', None)
        if test_budget is not None:
            test_budget -=  (2 * n) + num_initial_executions
            self.config['test_budget'] = test_budget
        
        pbar = tqdm.tqdm(total=test_budget) if test_budget else tqdm.tqdm()
        num_iterations = 0

        while True:
            if test_budget is not None and num_iterations >= test_budget:
                break

            input, acc_reward_input, generation, parent_seed_id = pool.select(self.rng) # 接收 seed_id
            new_generation = generation + 1
            mutant = self.mutate_validate(input, **kwargs)
            acc_reward_mutant, oracle, state_sequence, exec_time = self.mdp(mutant, policy)
            
            # [修改] 传入 parent_seed_id 作为当前样本的 seed_id
            self._save_observations(path, mutant, oracle, state_sequence, new_generation, seed_id=parent_seed_id)

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
                    sensitivity, _, _, _, _ = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy, generation=new_generation, seed_id=parent_seed_id, **kwargs)
                pool.add(mutant, acc_reward_mutant, coverage, sensitivity, oracle, generation=new_generation, seed_id=parent_seed_id) 

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
                    run_time=time.time(),
                    seed_id=parent_seed_id 
                )
            
            num_iterations += 1
            pbar.update(1)
        
        pbar.close()

        if path is not None:
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')

    def fuzzing_no_coverage(self, n: int, policy: Any = None, **kwargs):
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
        
        time_budget = kwargs.get('time_budget', None)
        test_budget = kwargs.get('test_budget', None)

        pbar = tqdm.tqdm(total=n, desc="Initialization")
        for i, state in enumerate(initial_inputs): # Enumerate seed_id
            sensitivity, acc_reward, oracle, state_sequence, exec_time = self.sentivity(state, policy=policy, generation=0, seed_id=i, **kwargs)
            # [修改] 传入 seed_id
            self._save_observations(path, state, oracle, state_sequence, 0, seed_id=i)
            pool.add(state, acc_reward, 0, sensitivity, oracle, generation=0, seed_id=i) # Pass seed_id

            if self.logger is not None:
                self.logger.log(
                    input=state,
                    oracle=oracle,
                    reward=acc_reward,
                    episode_length=len(state_sequence),
                    sensitivity=sensitivity,
                    Generation=0,
                    test_exec_time=exec_time,
                    run_time=time.time(),
                    seed_id=i 
                )
            if oracle:
                pool.add_crash(state)
            pbar.update(1)
        pbar.close()

        num_iterations = 0
        main_loop_start_time = time.time()
        
        if time_budget is not None:
            print(f"Fuzzing started with time budget: {time_budget} seconds.")
            pbar = tqdm.tqdm(desc="Fuzzing (Time-based)")
        else:
            if test_budget is not None:
                test_budget -= (2 * n)
            else:
                test_budget = 1000
            pbar = tqdm.tqdm(total=test_budget, desc="Fuzzing (Iter-based)")

        while True:
            current_time = time.time()
            elapsed_time = current_time - main_loop_start_time
            
            if time_budget is not None:
                if elapsed_time >= time_budget:
                    print(f"Time budget ({time_budget}s) reached. Stopping.")
                    break
                pbar.set_postfix({"elapsed": f"{elapsed_time:.1f}s"})
            elif test_budget is not None:
                if num_iterations >= test_budget:
                    break

            input, acc_reward_input, generation, parent_seed_id = pool.select(self.rng) # Receive seed_id
            new_generation = generation + 1
            mutant = self.mutate_validate(input, **kwargs)
            acc_reward_mutant, oracle, state_sequence, exec_time = self.mdp(mutant, policy)
            
            # [修改] 传入 parent_seed_id
            self._save_observations(path, mutant, oracle, state_sequence, new_generation, seed_id=parent_seed_id)

            sensitivity = None
            crash_time = None
            if oracle:
                crash_time = elapsed_time
                pool.add_crash(mutant)
            elif acc_reward_mutant < acc_reward_input:
                if local_sensitivity:
                    sensitivity = self.local_sensitivity(input, mutant, acc_reward_input, acc_reward_mutant)
                else:
                    sensitivity, _, _, _, _ = self.sentivity(mutant, acc_reward=acc_reward_mutant, policy=policy, generation=new_generation, seed_id=parent_seed_id, **kwargs)
                pool.add(mutant, acc_reward_mutant, 0, sensitivity, oracle, generation=new_generation, seed_id=parent_seed_id)

            if self.logger is not None:
                self.logger.log(
                    input=mutant,
                    oracle=oracle,
                    reward=acc_reward_mutant,
                    episode_length=len(state_sequence),
                    sensitivity=sensitivity,
                    Generation=new_generation,
                    test_exec_time=exec_time,
                    run_time=current_time,
                    crash_time=crash_time,
                    seed_id=parent_seed_id 
                )

            num_iterations += 1
            pbar.update(1)

        pbar.close()
        
        if path is not None:
            self.save_configuration(path)
            np.savetxt(path + '_selected.txt', pool.selected, fmt='%1.0f', delimiter=',')

    def random_testing(self, n: int = None, time_budget: float = None, policy: Any = None, path: str = 'logs', **kwargs):
        '''RT baseline with support for both iteration-based and time-based budget.'''
        if kwargs.get('exp_name', None) is not None:
            self.config['use_case'] = kwargs['exp_name']
        check_redundant_input = kwargs.get('check_redundant_input', True)

        self.config['name'] = 'RT'
        if time_budget is not None:
            self.config['time_budget'] = time_budget
            pbar = tqdm.tqdm(desc="RT (Time-based)")
        else:
            n = n if n is not None else 1000
            self.config['test_budget'] = n
            pbar = tqdm.tqdm(total=n, desc="RT (Iter-based)")
        
        if path is not None:
            self.logger = FuzzerLogger(path + '_logs.txt')
            self.logger.write_columns()
        else:
            self.logger = None
            
        i = 0
        start_time = time.time()

        while True:
            # Check stopping conditions
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if time_budget is not None:
                if elapsed_time >= time_budget:
                    print(f"Time budget ({time_budget}s) reached. Stopping.")
                    break
                pbar.set_postfix({"elapsed": f"{elapsed_time:.1f}s"})
            elif n is not None:
                if i >= n:
                    break

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
                
                # Crash detection time (relative to start)
                crash_time = elapsed_time if oracle else None
                
                # [修改] 显式传递 None 或不修改 (RT无血缘)
                self._save_observations(path, random_input, oracle, state_sequence, 0, seed_id=None)

                if self.logger is not None:
                    self.logger.log(
                        input=random_input,
                        oracle=oracle,
                        reward=acc_reward,
                        episode_length=len(state_sequence),
                        Generation=0,
                        test_exec_time=exec_time,
                        run_time=current_time,
                        crash_time=crash_time,
                        seed_id=None 
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