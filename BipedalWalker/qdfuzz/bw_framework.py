import json
import os
import time
import torch
import tqdm
import numpy as np
import pandas as pd

from stable_baselines3.common.base_class import BaseAlgorithm
from typing import List, Optional

from bw_common import load_model, EXPERT_INDICES, execute_policy, get_edges
from common import compute_cell, EXPERIMENT_SEEDS

class Framework():
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        self.version = 'random'
        self.rand_seed = rand_seed
        self.rng: np.random.Generator = np.random.default_rng(rand_seed)
        self.creation_time = time.time()

        self.loaded = False
        self.has_init = False
        self.init_budget = None

        self.granularity = cell_granularity
        self.descriptors: np.ndarray = np.array(descriptors)
        # 2d behavior spaces
        assert len(self.descriptors) == 2
        assert all(self.descriptors < 12) and all(self.descriptors >= 0)

        # as indices
        self.last_cell_selected = None
        self.last_cell_updated = None

        # data structure consists of a list of cells (list of integers) and a list of list of test results
        self.cells: list[list[int]] = []
        # a 6-tuple: (input, performance, oracle result, behavior, mutation_count, elapsed_time)
        self.cells_data: list[list[tuple[np.ndarray, float, bool, np.ndarray, int, float]]] = []

        self.config = {
            'rand_seed': self.rand_seed,
            'cell_granularity': self.granularity,
            'descriptors': self.descriptors.tolist(),
            'use_case': 'Bipedal Walker'
        }

        # kwargs (name to include in the experimental configuration etc.)
        self.name = kwargs.get('name')
        if self.name is not None:
            self.config['name'] = self.name
        else:
            self.config['name'] = self.version

        try:
            index = EXPERT_INDICES.index(self.descriptors.tolist())
            self.config['use_case'] = f'Bipedal Walker {index}'
        except ValueError:
            pass


    def save_configuration(self, filepath: str):
        '''
        Saves the configuration of the object.
        '''
        if not filepath.endswith('config'):
            filepath += '_config'
        f = open(f'{filepath}.json', 'w')
        f.write(json.dumps(self.config))
        f.close()


    def save_random_state(self, filepath: str):
        '''Saves the state of the BitGenerator instance (of the Generator).'''
        f = open(f'{filepath}_state.json', 'w')
        f.write(json.dumps(self.rng.bit_generator.state))
        f.close()
        return self.rng.bit_generator.state


    def save_state(self, filepath: str):
        '''
        Saves the current state of the framework to possibly resume execution.
        '''
        cell_dfs = []
        
        # 定义基础列名
        base_columns = ['score', 'is_faulty', 'cell_index'] + [f'cell{i}' for i in range(2)] + [f'behavior{i}' for i in range(2)]
        all_columns = base_columns + ['input', 'mutation_count', 'elapsed_time']

        for i, cell_data in enumerate(self.cells_data):
            # 准备数据，显式将 input 转为 JSON 字符串
            records = []
            for item in cell_data:
                _input, score, is_faulty, behavior, mutation_count, elapsed_time = item
                # 构建一行数据
                record = [
                    score, 
                    is_faulty, 
                    i, 
                    self.cells[i][0], self.cells[i][1], # cell 坐标
                    behavior[0], behavior[1],           # behavior 特征
                    json.dumps(_input.tolist()),        # input (序列化为字符串)
                    mutation_count,
                    elapsed_time
                ]
                records.append(record)

            cell_dfs.append(
                pd.DataFrame.from_records(
                    data=records,
                    columns=all_columns
                    )
                )
        
        if cell_dfs:
            pd.concat(cell_dfs, ignore_index=True).to_csv(f'{filepath}_data.csv', index=0)
            print(f"Saved state to {filepath}_data.csv")
        else:
            print("No data in cells_data to save.")
            
        self.save_random_state(filepath)
        self.save_configuration(filepath)


    def load_configuration(self, filepath: str):
        '''Loads and sets the configuration attribute of the instance.'''
        if not filepath.endswith('config'):
            filepath += '_config'
        f = open(f'{filepath}.json', 'r')
        self.config = json.load(f)
        f.close()


    def load_random_state(self, filepath: str):
        '''Loads and sets the state of BitGenerator instance (of the Generator).'''
        if not filepath.endswith('state'):
            filepath += '_state'
        f = open(f'{filepath}.json', 'r')
        self.rng.bit_generator.state = json.load(f)
        f.close()


    def load_state(self, filepath: str):
        '''
        Loads a state of an instance to resume testing and returns the number of test cases loaded.
        '''
        df_fp = f'{filepath}_data.csv'

        assert os.path.exists(df_fp), 'file is missing.'
        self.cells = []
        self.cells_data = []

        df = pd.read_csv(df_fp)

        cell_cols = [c for c in df.columns.to_list() if c.startswith('cell') and 'index' not in c]
        behavior_cols = [c for c in df.columns.to_list() if c.startswith('behavior')]

        assert len(cell_cols) > 0, "CSV 中未找到 Cell 列"
        assert len(behavior_cols) > 0, "CSV 中未找到 Behavior 列"

        has_elapsed_time = 'elapsed_time' in df.columns
        has_mutation_count = 'mutation_count' in df.columns

        for i, row in df.iterrows():
            cell = row[cell_cols].astype(int).tolist()
            performance = row['score']
            is_faulty = row['is_faulty']
            behavior = row[behavior_cols].values
            
            try:
                input_vec = np.array(json.loads(row['input']), dtype=int)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Failed to parse input at row {i}, skipping.")
                continue
            
            mutation_count = int(row['mutation_count']) if has_mutation_count else 0
            elapsed_time = float(row['elapsed_time']) if has_elapsed_time else 0.0
            
            self.update_cell(cell, input_vec, performance, is_faulty, np.array(behavior), mutation_count, elapsed_time)

        self.load_random_state(filepath)
        self.load_configuration(filepath)
        self.loaded = True
        return len(df)


    def select_input(self, index: int):
        '''
        Samples from the indexed cell the next input.
        Returns both input and its mutation_count.
        '''
        input_index: int = self.rng.integers(0, len(self.cells_data[index]))
        selected_data = self.cells_data[index][input_index]
        return selected_data[0], selected_data[4]


    def select_cell(self):
        '''Selects the cell for the next search iteration.'''
        return int(self.rng.integers(0, len(self.cells)))


    def update_cell(self, cell: List[int], input: np.ndarray, performance: float, is_faulty: bool, behavior: np.ndarray, mutation_count: int, elapsed_time: float):
        '''
        Records the execution result to the corresponding cell.
        '''
        index = None
        try:
            index = self.cells.index(cell)
            self.cells_data[index].append((input, performance, is_faulty, behavior, mutation_count, elapsed_time))
        except ValueError:
            self.cells.append(cell)
            self.cells_data.append([(input, performance, is_faulty, behavior, mutation_count, elapsed_time)])
        finally:
            assert len(self.cells) == len(self.cells_data), 'inconsistent cells and cells_data lists!'
            self.last_cell_updated = index if index is not None else (len(self.cells) - 1)
        return self.last_cell_updated


    def mutate(self, input: np.ndarray) -> np.ndarray:
        mutation = self.rng.choice(2, 15, p=[0.9, 0.1])
        if np.sum(mutation) == 0:
            mutation[0] = 1
        mutated_input = input + mutation
        mutated_input = np.remainder(mutated_input, 4)
        mutated_input = np.clip(mutated_input, 1, 3)
        return mutated_input


    def _check_budget(self, start_time: float, current_executions: int, time_budget_hours: Optional[float], execution_budget: Optional[int]) -> bool:
        """
        Helper to check if any budget is exceeded. Returns True if budget is NOT reached (continue), False if reached (stop).
        """
        if execution_budget is not None and current_executions >= execution_budget:
            print(f"Execution budget ({execution_budget}) reached.")
            return False
        
        if time_budget_hours is not None:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds >= time_budget_hours * 3600:
                print(f"Time budget ({time_budget_hours}h) reached.")
                return False
        
        return True


    def test_policy(self, model: BaseAlgorithm,
                    env_seed: int,
                    init_budget: int,
                    results_fp: str,
                    time_budget_hours: Optional[float] = None, # 可选时间
                    execution_budget: Optional[int] = None,    # 可选次数
                    disable_pbar: bool = False):

        if time_budget_hours is None and execution_budget is None:
            raise ValueError("At least one budget (time_budget_hours or execution_budget) must be provided.")

        self.config['time_budget_hours'] = time_budget_hours
        self.config['execution_budget'] = execution_budget
        self.init_budget = init_budget
        self.config['init_budget'] = self.init_budget
        self.config['env_seed'] = env_seed

        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp

        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        behaviors_buffer = open(f'{filepath}_behaviors.txt', 'w', buffering=1)
        final_states_buffer = open(f'{filepath}_final_states.txt', 'w', buffering=1)
        inputs_buffer = open(f'{filepath}_inputs.txt', 'w', buffering=1)
        cells_buffer = open(f'{filepath}_cells.txt', 'w', buffering=1)
        logs_buffer = open(f'{filepath}_logs.txt', 'w', buffering=1)

        print(f'Starting test_policy. Time Budget: {time_budget_hours}h, Execution Budget: {execution_budget}')

        inputs: List[np.ndarray] = []
        behaviors = []
        final_states: List[np.ndarray] = []
        acc_rewards: List[float] = []
        oracles: List[bool] = []
        
        testing_start_time = time.time()
        execution_times = []
        n_executions = 0 

        print("Starting initialization phase...")
        for _ in tqdm.tqdm(range(init_budget), disable=disable_pbar):
            # 检查预算
            if not self._check_budget(testing_start_time, n_executions, time_budget_hours, execution_budget):
                print("Budget reached during initialization.")
                break

            input: np.ndarray = self.rng.integers(low=1, high=4, size=15)

            t0 = time.time()
            episode_reward, oracle, behavior, fs, _ = execute_policy(input, model, env_seed, self.descriptors)
            t1 = time.time()
            execution_times.append(t1 - t0)

            inputs.append(input)
            behaviors.append(behavior)
            final_states.append(fs)
            acc_rewards.append(episode_reward)
            oracles.append(oracle)
            n_executions += 1
        
        if not inputs:
            print("No inputs generated.")
            behaviors_buffer.close()
            inputs_buffer.close()
            cells_buffer.close()
            logs_buffer.close()
            final_states_buffer.close()
            return

        behaviors = np.array(behaviors)

        self.xedges, self.yedges = get_edges(env_seed, self.descriptors)
        self.config['xedges'] = list(self.xedges)
        self.config['yedges'] = list(self.xedges)

        for i in range(len(inputs)): 
            behavior = behaviors[i]
            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()
            mutated_input_index = self.update_cell(cell, inputs[i], acc_rewards[i], oracles[i], behavior, 0, 0.0)
            print(f'episode_reward: {acc_rewards[i]}, oracle: {float(oracles[i])}, cell_selected_index: -1, cell_updated_index: {mutated_input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, inputs[i].reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
            np.savetxt(final_states_buffer, final_states[i].reshape(1, -1), delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')

        # 配置进度条
        if execution_budget is not None:
            pbar = tqdm.tqdm(total=execution_budget, initial=n_executions, disable=disable_pbar)
            pbar.set_description(f"Fuzzing (max {execution_budget} execs)")
        else:
            pbar = tqdm.tqdm(disable=disable_pbar)
            pbar.set_description(f"Fuzzing (max {time_budget_hours}h)")

        print("Starting fuzzing loop...")
        fuzzing_start_time = time.time()

        while self._check_budget(testing_start_time, n_executions, time_budget_hours, execution_budget):
            cell_index = self.select_cell()
            self.last_cell_selected = cell_index
            input, parent_mutation_count = self.select_input(cell_index)

            mutated_input = self.mutate(input)
            t0 = time.time()
            episode_reward, oracle, behavior, fs, _ = execute_policy(mutated_input, model, env_seed, self.descriptors)
            t1 = time.time()
            execution_times.append(t1 - t0)
            
            n_executions += 1 

            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()
            new_mutation_count = parent_mutation_count + 1
            elapsed_time = time.time() - fuzzing_start_time
            
            mutated_input_index = self.update_cell(cell, mutated_input, episode_reward, oracle, behavior, new_mutation_count, elapsed_time)
            print(f'episode_reward: {episode_reward}, oracle: {float(oracle)}, cell_selected_index: {cell_index}, cell_updated_index: {mutated_input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, mutated_input.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
            np.savetxt(final_states_buffer, fs.reshape(1, -1), delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
            
            pbar.update(1)

        print("Stopping test_policy.")
        testing_end_time = time.time()
        self.config['testing_start_time'] = testing_start_time
        self.config['testing_end_time'] = testing_end_time
        self.config['testing_time'] = testing_end_time - testing_start_time
        self.config['total_execution_time'] = sum(execution_times)
        self.config['fuzzing_start_time'] = fuzzing_start_time
        self.config['total_executions'] = n_executions

        pbar.close()
        behaviors_buffer.close()
        inputs_buffer.close()
        cells_buffer.close()
        logs_buffer.close()
        final_states_buffer.close()
        self.save_state(filepath)


    def random_testing(self, model: BaseAlgorithm,
                    env_seed: int,
                    results_fp: str,
                    time_budget_hours: Optional[float] = None,
                    execution_budget: Optional[int] = None,
                    disable_pbar: bool = False):
        '''Random testing loop baseline.'''
        
        if time_budget_hours is None and execution_budget is None:
            raise ValueError("At least one budget must be provided.")

        self.config['time_budget_hours'] = time_budget_hours
        self.config['execution_budget'] = execution_budget
        self.config['env_seed'] = env_seed

        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp
        
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        behaviors_buffer = open(f'{filepath}_behaviors.txt', 'w', buffering=1)
        final_states_buffer = open(f'{filepath}_final_states.txt', 'w', buffering=1)
        inputs_buffer = open(f'{filepath}_inputs.txt', 'w', buffering=1)
        cells_buffer = open(f'{filepath}_cells.txt', 'w', buffering=1)
        logs_buffer = open(f'{filepath}_logs.txt', 'w', buffering=1)

        print(f'Starting random_testing. Time Budget: {time_budget_hours}h, Execution Budget: {execution_budget}')

        self.xedges, self.yedges = get_edges(env_seed, self.descriptors)
        self.config['xedges'] = list(self.xedges)
        self.config['yedges'] = list(self.xedges)

        execution_times = []
        n_executions = 0 
        start_time = time.time()
        
        if execution_budget is not None:
            pbar = tqdm.tqdm(total=execution_budget, disable=disable_pbar)
        else:
            pbar = tqdm.tqdm(disable=disable_pbar)

        while self._check_budget(start_time, n_executions, time_budget_hours, execution_budget):
            input: np.ndarray = self.rng.integers(low=1, high=4, size=15)
            t0 = time.time()
            episode_reward, oracle, behavior, fs, _ = execute_policy(input, model, env_seed, self.descriptors)
            t1 = time.time()
            execution_times.append(t1 - t0)
            
            n_executions += 1 

            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()
            elapsed_time = time.time() - start_time

            input_index = self.update_cell(cell, input, episode_reward, oracle, behavior, 0, elapsed_time)
            print(f'episode_reward: {episode_reward}, oracle: {float(oracle)}, cell_selected_index: -1, cell_updated_index: {input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, input.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
            np.savetxt(final_states_buffer, fs.reshape(1, -1), delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
            
            pbar.update(1)

        print("Stopping random_testing.")
        testing_end_time = time.time()
        self.config['testing_start_time'] = start_time
        self.config['testing_end_time'] = testing_end_time
        self.config['testing_time'] = testing_end_time - start_time
        self.config['total_execution_time'] = sum(execution_times)
        self.config['total_executions'] = n_executions

        pbar.close()
        behaviors_buffer.close()
        inputs_buffer.close()
        cells_buffer.close()
        logs_buffer.close()
        final_states_buffer.close()
        self.save_state(filepath)


    def novelty_search(self, model: BaseAlgorithm,
                    env_seed: int,
                    pop_size: int,
                    k: int,
                    nov_threshold: float,
                    results_fp: str,
                    time_budget_hours: Optional[float] = None,
                    execution_budget: Optional[int] = None,
                    disable_pbar: bool = False):
        '''Does not use cached data anymore.'''

        if time_budget_hours is None and execution_budget is None:
            raise ValueError("At least one budget must be provided.")

        self.config['pop_size'] = pop_size
        self.config['time_budget_hours'] = time_budget_hours
        self.config['execution_budget'] = execution_budget
        self.config['env_seed'] = env_seed
        self.config['nov_threshold'] = nov_threshold
        self.config['k'] = k

        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp
        
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        behaviors_buffer = open(f'{filepath}_behaviors.txt', 'w', buffering=1)
        final_states_buffer = open(f'{filepath}_final_states.txt', 'w', buffering=1)
        inputs_buffer = open(f'{filepath}_inputs.txt', 'w', buffering=1)
        cells_buffer = open(f'{filepath}_cells.txt', 'w', buffering=1)
        logs_buffer = open(f'{filepath}_logs.txt', 'w', buffering=1)

        testing_start_time = time.time()
        n_executions = 0
        
        print(f'Starting novelty_search. Time Budget: {time_budget_hours}h, Execution Budget: {execution_budget}')

        self.xedges, self.yedges = get_edges(env_seed, self.descriptors)
        self.config['xedges'] = list(self.xedges)
        self.config['yedges'] = list(self.xedges)

        def record(input: np.ndarray, reward: float, oracle: bool, behavior: np.ndarray, final_state: np.ndarray, mutation_count: int, elapsed_time: float) -> None:
            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()
            updated_cell_index = self.update_cell(cell, input, reward, oracle, behavior, mutation_count, elapsed_time)
            print(f'episode_reward: {reward}, oracle: {float(oracle)}, cell_updated_index: {updated_cell_index}, nb_cells: {len(self.cells)}', file=logs_buffer)
            np.savetxt(inputs_buffer, input.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
            np.savetxt(final_states_buffer, final_state.reshape(1, -1), delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
        
        def evaluate(individuals: np.ndarray, mutation_counts: np.ndarray, loop_start_time: float = None) -> np.ndarray:
            nonlocal n_executions 
            behaviors = []
            for i, ind in enumerate(individuals):
                if not self._check_budget(testing_start_time, n_executions, time_budget_hours, execution_budget):
                    break 
                
                r, o, b, fs, _ = execute_policy(ind, model, env_seed, self.descriptors, 300)
                n_executions += 1
                
                if loop_start_time is None:
                    e_time = 0.0
                else:
                    e_time = time.time() - loop_start_time

                record(ind, r, o, b, fs, mutation_counts[i], e_time)
                behaviors.append(b)
            return np.array(behaviors)
        
        def mutate(inputs: np.ndarray):
            mutants = [self.mutate(input) for input in inputs]
            return np.array(mutants)

        ns_logs_buffer = open(f'{filepath}_ns_logs.txt', 'w', buffering=1)
        nov_scores_buffer = open(f'{filepath}_nov_scores.txt', 'w', buffering=1)
        
        from novelty_search import NoveltyArchive
        print("Starting initial population evaluation...")
        pop = self.rng.integers(low=1, high=4, size=(pop_size, 15))
        pop_mutation_counts = np.zeros(pop_size, dtype=int)
        
        pop_behaviors = evaluate(pop, pop_mutation_counts, loop_start_time=None)
        
        if not pop_behaviors.any() or not self._check_budget(testing_start_time, n_executions, time_budget_hours, execution_budget):
             print("Budget reached or no behaviors generated.")
             ns_logs_buffer.close()
             nov_scores_buffer.close()
             behaviors_buffer.close()
             inputs_buffer.close()
             cells_buffer.close()
             logs_buffer.close()
             final_states_buffer.close()
             return

        nov_archive = NoveltyArchive(pop_behaviors, k, nov_threshold)
        pop_nov_scores = nov_archive.score(pop_behaviors)
        [np.savetxt(nov_scores_buffer, s.reshape(1, -1), delimiter=',') for s in pop_nov_scores]
        
        print(f'iteration: 0, archive_size: {nov_archive.size()}, archive_sparseness: {nov_archive.archive_sparseness():0.5f}', file=ns_logs_buffer)
        
        i = 1
        if execution_budget is not None:
             pbar = tqdm.tqdm(total=execution_budget, initial=n_executions, disable=disable_pbar)
        else:
             pbar = tqdm.tqdm(disable=disable_pbar)

        print("Starting Novelty Search loop...")
        ns_start_time = time.time()

        while self._check_budget(testing_start_time, n_executions, time_budget_hours, execution_budget):
            offspring = mutate(pop)
            offspring_mutation_counts = pop_mutation_counts + 1
            
            prev_executions = n_executions
            offspring_behaviors = evaluate(offspring, offspring_mutation_counts, loop_start_time=ns_start_time)
            executions_diff = n_executions - prev_executions
            pbar.update(executions_diff)
            
            if not offspring_behaviors.any():
                print("Budget reached during offspring evaluation.")
                break

            offspring_nov_scores = nov_archive.score(offspring_behaviors, pop_behaviors)

            joined_pop = np.vstack([pop, offspring[:len(offspring_behaviors)]]) 
            joined_scores = np.hstack([pop_nov_scores, offspring_nov_scores])
            joined_mutation_counts = np.hstack([pop_mutation_counts, offspring_mutation_counts[:len(offspring_behaviors)]])
            
            median_score = np.median(joined_scores)

            _updated, _offspring_indices = nov_archive.update3(offspring_behaviors)

            mask = (joined_scores >= median_score)

            pop = joined_pop[mask].copy()
            pop_behaviors = np.vstack([pop_behaviors, offspring_behaviors])[mask]
            pop_mutation_counts = joined_mutation_counts[mask]
            
            pop_nov_scores = nov_archive.score(pop_behaviors)
            if len(pop) > pop_size:
                pop = pop[:pop_size]
                pop_behaviors = pop_behaviors[:pop_size]
                pop_nov_scores = pop_nov_scores[:pop_size]
                pop_mutation_counts = pop_mutation_counts[:pop_size]

            [np.savetxt(nov_scores_buffer, s.reshape(1, -1), delimiter=',') for s in pop_nov_scores]
            print(f'iteration: {i}, archive_size: {nov_archive.size()}, archive_sparseness: {nov_archive.archive_sparseness():0.5f}', file=ns_logs_buffer)
            
            i += 1

        print("Stopping novelty_search.")
        pbar.close()
        ns_logs_buffer.close()
        nov_scores_buffer.close()
        
        behaviors_buffer.close()
        inputs_buffer.close()
        cells_buffer.close()
        logs_buffer.close()
        final_states_buffer.close()
        self.save_state(filepath)

#TODO: this version can actually only keep the best performing input per cell (since all execution data is recorded during testing)
class MAPElitesFramework(Framework):
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        if kwargs.get('name') is None:
            kwargs['name'] = 'MAP-Elites'
        super().__init__(rand_seed, cell_granularity, descriptors, **kwargs)


    def select_input(self, index: int):
        '''
        Returns both input and its mutation_count.
        '''
        scores = list(map(lambda x: x[1], self.cells_data[index]))
        best_performer_index = int(np.argmin(scores))
        selected_data = self.cells_data[index][best_performer_index]
        return selected_data[0], selected_data[4]


if __name__ == '__main__':
    torch.set_num_threads(1)
    main_seed = 2021
    env_seed = 0
    model = load_model()

    # --- 配置区域 ---
    # 你可以在这里同时设置时间预算(小时)和执行次数预算。
    # 程序会在任一限制达到时停止。设置为 None 则忽略该限制。
    
    TIME_BUDGET_HOURS = None       # 例如: 12 小时
    EXECUTION_BUDGET = 5000     # 例如: 20000 次执行
    
    # ----------------
    
    init_budget = 1000
    cell_granularity = 50

    population_size = 100
    k = 3
    novelty_threshold = 0.005

    results_fp = 'results/bw'
    if not os.path.exists(results_fp):
        os.makedirs(results_fp)

    for seed in EXPERIMENT_SEEDS:
        print(f'Seed {seed} starts.')
        for expert_indices in EXPERT_INDICES:
            print(f"--- Running MAP-Elites ---")
            f = MAPElitesFramework(seed, cell_granularity, descriptors=expert_indices, name='MAP-Elites')
            
            # 使用新的参数接口调用
            f.test_policy(
                model=model, 
                env_seed=env_seed, 
                init_budget=init_budget, 
                results_fp=results_fp,
                time_budget_hours=TIME_BUDGET_HOURS,    # 传入时间预算
                execution_budget=EXECUTION_BUDGET       # 传入执行预算
            )
            
        print(f'Experts done.')