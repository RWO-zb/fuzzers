import json
import os
import time
import torch
import tqdm
import numpy as np
import pandas as pd

from stable_baselines3.common.base_class import BaseAlgorithm
from typing import List

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
        # MODIFIED: 移除了 test_budget
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
        # MODIFIED: a 6-tuple: (input, performance, oracle result, behavior, mutation_count, elapsed_time)
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
        This lets us know what BS has been used, which can be handy for organizing the results and to compare to MDPFuzz.
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
        MODIFIED: Saves the input vector as a JSON string to ensure correct CSV serialization.
        '''
        cell_dfs = []
        
        # 定义基础列名
        base_columns = ['score', 'is_faulty', 'cell_index'] + [f'cell{i}' for i in range(2)] + [f'behavior{i}' for i in range(2)]
        
        # MODIFIED: 定义包含 'input', 'mutation_count', 'elapsed_time' 的列
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
        
        if cell_dfs: # 确保列表不为空
            pd.concat(cell_dfs, ignore_index=True).to_csv(f'{filepath}_data.csv', index=0)
            print(f"Saved state to {filepath}_data.csv")
        else:
            print("No data in cells_data to save.")
            
        # saves the random state
        self.save_random_state(filepath)
        # saves the configuration
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
        MODIFIED: Loads the input vector from a JSON string column.
        '''
        df_fp = f'{filepath}_data.csv'

        assert os.path.exists(df_fp), 'file is missing.'
        self.cells = []
        self.cells_data = []

        df = pd.read_csv(df_fp)

        # 确定单元格和行为列
        cell_cols = [c for c in df.columns.to_list() if c.startswith('cell') and 'index' not in c]
        behavior_cols = [c for c in df.columns.to_list() if c.startswith('behavior')]

        assert len(cell_cols) > 0, "CSV 中未找到 Cell 列"
        assert len(behavior_cols) > 0, "CSV 中未找到 Behavior 列"

        # 检查是否存在 elapsed_time 和 mutation_count 列（为了兼容旧数据）
        has_elapsed_time = 'elapsed_time' in df.columns
        has_mutation_count = 'mutation_count' in df.columns

        for i, row in df.iterrows():
            # MODIFIED: 从行中提取数据
            cell = row[cell_cols].astype(int).tolist()
            performance = row['score']
            is_faulty = row['is_faulty']
            behavior = row[behavior_cols].values
            
            # MODIFIED: 使用 json.loads 将 'input' 列的字符串解析回列表,然后转为 numpy 数组
            try:
                input_vec = np.array(json.loads(row['input']), dtype=int)
            except (json.JSONDecodeError, TypeError):
                # 兼容可能的旧格式或出错情况
                print(f"Warning: Failed to parse input at row {i}, skipping.")
                continue
            
            # MODIFIED: 加载 mutation_count
            mutation_count = int(row['mutation_count']) if has_mutation_count else 0

            # MODIFIED: 加载 elapsed_time
            elapsed_time = float(row['elapsed_time']) if has_elapsed_time else 0.0
            
            # MODIFIED: 调用新的 update_cell
            self.update_cell(cell, input_vec, performance, is_faulty, np.array(behavior), mutation_count, elapsed_time)

        self.load_random_state(filepath)
        self.load_configuration(filepath)
        self.loaded = True
        return len(df)


    def select_input(self, index: int):
        '''
        Samples from the indexed cell the next input.
        MODIFIED: Returns both input and its mutation_count.
        '''
        input_index: int = self.rng.integers(0, len(self.cells_data[index]))
        
        selected_data = self.cells_data[index][input_index]
        # 返回 input (索引 0) 和 mutation_count (索引 4)
        # 索引 5 是 elapsed_time，不需要在这里返回
        return selected_data[0], selected_data[4]


    def select_cell(self):
        '''Selects the cell for the next search iteration.'''
        return int(self.rng.integers(0, len(self.cells)))


    def update_cell(self, cell: List[int], input: np.ndarray, performance: float, is_faulty: bool, behavior: np.ndarray, mutation_count: int, elapsed_time: float):
        '''
        Records the execution result to the corresponding cell.
        MODIFIED: Accepts and stores mutation_count and elapsed_time.
        It returns the index of the cell updated.
        '''
        index = None
        try:
            # index of the cell to update
            index = self.cells.index(cell)
            # MODIFIED: 存储 6 元素元组
            self.cells_data[index].append((input, performance, is_faulty, behavior, mutation_count, elapsed_time))
        except ValueError:
            self.cells.append(cell)
            # MODIFIED: 存储 6 元素元组
            self.cells_data.append([(input, performance, is_faulty, behavior, mutation_count, elapsed_time)])
        finally:
            # sanity checks
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


    def test_policy(self, model: BaseAlgorithm,
                    env_seed: int,
                    time_budget_hours: int, # MODIFIED: 接收小时预算
                    init_budget: int,
                    results_fp: str,
                    disable_pbar: bool = False):

        # MODIFIED: 移除了 test_budget
        self.config['time_budget_hours'] = time_budget_hours
        self.init_budget = init_budget
        self.config['init_budget'] = self.init_budget

        self.config['env_seed'] = env_seed


        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp

        # 确保目录存在
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        behaviors_buffer = open(f'{filepath}_behaviors.txt', 'w', buffering=1)
        final_states_buffer = open(f'{filepath}_final_states.txt', 'w', buffering=1)
        inputs_buffer = open(f'{filepath}_inputs.txt', 'w', buffering=1)
        cells_buffer = open(f'{filepath}_cells.txt', 'w', buffering=1)
        logs_buffer = open(f'{filepath}_logs.txt', 'w', buffering=1)

        # MODIFIED: 
        time_budget_seconds = time_budget_hours * 3600
        # MODIFIED: 移除了 executions_budget
        print(f'Time budget of {time_budget_hours} hours ({(time_budget_seconds / 60):.2f} minutes).')

        inputs: List[np.ndarray] = []
        behaviors = []
        final_states: List[np.ndarray] = []
        acc_rewards: List[float] = []
        oracles: List[bool] = []
        # MODIFIED: 在 init 循环之前启动总计时器
        testing_start_time = time.time()
        execution_times = []

        print("Starting initialization phase...")
        for _ in tqdm.tqdm(range(init_budget), disable=disable_pbar):
            # MODIFIED: 添加时间检查以允许提前退出 init
            if (time.time() - testing_start_time > time_budget_seconds):
                print("Time budget reached during initialization. Stopping.")
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
        
        # 确保即使 init 提前停止，代码也能继续
        if not inputs:
            print("No inputs generated, time budget may be too small.")
            behaviors_buffer.close()
            inputs_buffer.close()
            cells_buffer.close()
            logs_buffer.close()
            final_states_buffer.close()
            return # 提前退出

        behaviors = np.array(behaviors)

        self.xedges, self.yedges = get_edges(env_seed, self.descriptors)

        self.config['xedges'] = list(self.xedges)
        self.config['yedges'] = list(self.xedges)

        for i in range(len(inputs)): # MODIFIED: 使用 len(inputs) 应对提前退出的情况
            behavior = behaviors[i]
            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()
            # MODIFIED: 初始种子, mutation_count = 0, elapsed_time = 0.0 (表示在 Fuzzing Loop 之前)
            mutated_input_index = self.update_cell(cell, inputs[i], acc_rewards[i], oracles[i], behavior, 0, 0.0)
            print(f'episode_reward: {acc_rewards[i]}, oracle: {float(oracles[i])}, cell_selected_index: -1, cell_updated_index: {mutated_input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, inputs[i].reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
            np.savetxt(final_states_buffer, final_states[i].reshape(1, -1), delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')

        # MODIFIED: 移除了 start_time, nb_executions
        current_time = time.time()
        # MODIFIED: 移除了 pbar 的 total
        pbar = tqdm.tqdm(disable=disable_pbar)
        pbar.set_description(f"Fuzzing loop (running for {time_budget_hours}h total)")

        print("Starting fuzzing loop...")
        
        # MODIFIED: 记录主循环开始时间
        fuzzing_start_time = time.time()

        # MODIFIED: 循环条件只检查总时间
        while (current_time - testing_start_time < time_budget_seconds):
            cell_index = self.select_cell()
            self.last_cell_selected = cell_index
            # MODIFIED: select_input 现在返回 input 和 count
            input, parent_mutation_count = self.select_input(cell_index)

            mutated_input = self.mutate(input)
            t0 = time.time()
            episode_reward, oracle, behavior, fs, _ = execute_policy(mutated_input, model, env_seed, self.descriptors)
            t1 = time.time()
            execution_times.append(t1 - t0)
            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()

            # MODIFIED: 传递新的 count (父代 + 1)
            new_mutation_count = parent_mutation_count + 1
            
            # MODIFIED: 计算距离主循环开始的时间
            elapsed_time = time.time() - fuzzing_start_time
            
            mutated_input_index = self.update_cell(cell, mutated_input, episode_reward, oracle, behavior, new_mutation_count, elapsed_time)
            print(f'episode_reward: {episode_reward}, oracle: {float(oracle)}, cell_selected_index: {cell_index}, cell_updated_index: {mutated_input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, mutated_input.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
            np.savetxt(final_states_buffer, fs.reshape(1, -1), delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
            current_time = time.time()
            # MODIFIED: 移除了 nb_executions += 1
            pbar.update(1)

        print("Time budget reached. Stopping test_policy.")
        testing_end_time = time.time()
        self.config['testing_start_time'] = testing_start_time
        self.config['testing_end_time'] = testing_end_time
        self.config['testing_time'] = testing_end_time - testing_start_time
        self.config['total_execution_time'] = sum(execution_times)
        # MODIFIED: 也可以记录 fuzzing 开始时间
        self.config['fuzzing_start_time'] = fuzzing_start_time

        pbar.close()
        behaviors_buffer.close()
        inputs_buffer.close()
        cells_buffer.close()
        logs_buffer.close()
        final_states_buffer.close()
        self.save_state(filepath)


    def random_testing(self, model: BaseAlgorithm,
                    env_seed: int,
                    time_budget_hours: int, # MODIFIED: 接收小时预算
                    results_fp: str,
                    disable_pbar: bool = False):
        '''Random testing loop baseline.'''
        # MODIFIED:
        self.config['time_budget_hours'] = time_budget_hours
        self.config['env_seed'] = env_seed


        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp
        
        # 确保目录存在
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        behaviors_buffer = open(f'{filepath}_behaviors.txt', 'w', buffering=1)
        final_states_buffer = open(f'{filepath}_final_states.txt', 'w', buffering=1)
        inputs_buffer = open(f'{filepath}_inputs.txt', 'w', buffering=1)
        cells_buffer = open(f'{filepath}_cells.txt', 'w', buffering=1)
        logs_buffer = open(f'{filepath}_logs.txt', 'w', buffering=1)

        # MODIFIED:
        time_budget_seconds = time_budget_hours * 3600
        print(f'Time budget of {time_budget_hours} hours ({(time_budget_seconds / 60):.2f} minutes).')


        self.xedges, self.yedges = get_edges(env_seed, self.descriptors)

        self.config['xedges'] = list(self.xedges)
        self.config['yedges'] = list(self.xedges)

        execution_times = []

        start_time = time.time()
        current_time = time.time()
        # MODIFIED: 移除了 nb_executions
        pbar = tqdm.tqdm(disable=disable_pbar) # MODIFIED: 移除了 total
        pbar.set_description(f"Random testing (running for {time_budget_hours}h total)")

        # MODIFIED: 循环条件只检查总时间
        while (current_time - start_time < time_budget_seconds):
            input: np.ndarray = self.rng.integers(low=1, high=4, size=15)
            t0 = time.time()
            episode_reward, oracle, behavior, fs, _ = execute_policy(input, model, env_seed, self.descriptors)
            t1 = time.time()
            execution_times.append(t1 - t0)
            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()

            # MODIFIED: 计算时间
            elapsed_time = time.time() - start_time

            # MODIFIED: 随机测试, mutation_count = 0
            input_index = self.update_cell(cell, input, episode_reward, oracle, behavior, 0, elapsed_time)
            print(f'episode_reward: {episode_reward}, oracle: {float(oracle)}, cell_selected_index: -1, cell_updated_index: {input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, input.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
            np.savetxt(final_states_buffer, fs.reshape(1, -1), delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
            current_time = time.time()
            # MODIFIED: 移除了 nb_executions += 1
            pbar.update(1)

        print("Time budget reached. Stopping random_testing.")
        testing_end_time = time.time()
        self.config['testing_start_time'] = start_time
        self.config['testing_end_time'] = testing_end_time
        self.config['testing_time'] = testing_end_time - start_time
        self.config['total_execution_time'] = sum(execution_times)

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
                    # MODIFIED: 移除了 nb_iterations
                    k: int,
                    nov_threshold: float,
                    time_budget_hours: int, # MODIFIED: 接收小时预算
                    results_fp: str,
                    disable_pbar: bool = False):
        '''Does not use cached data anymore.'''

        self.config['pop_size'] = pop_size
        # MODIFIED: 移除了 nb_iterations 和 test_budget
        self.config['time_budget_hours'] = time_budget_hours
        self.config['env_seed'] = env_seed
        self.config['nov_threshold'] = nov_threshold
        self.config['k'] = k

        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp
        
        # 确保目录存在
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # to collect the data during the search, i.e., every model execution
        behaviors_buffer = open(f'{filepath}_behaviors.txt', 'w', buffering=1)
        final_states_buffer = open(f'{filepath}_final_states.txt', 'w', buffering=1)
        inputs_buffer = open(f'{filepath}_inputs.txt', 'w', buffering=1)
        cells_buffer = open(f'{filepath}_cells.txt', 'w', buffering=1)
        logs_buffer = open(f'{filepath}_logs.txt', 'w', buffering=1)

        # MODIFIED: 添加总计时器
        testing_start_time = time.time()
        time_budget_seconds = time_budget_hours * 3600
        print(f'Time budget of {time_budget_hours} hours ({(time_budget_seconds / 60):.2f} minutes).')

        self.xedges, self.yedges = get_edges(env_seed, self.descriptors)

        self.config['xedges'] = list(self.xedges)
        self.config['yedges'] = list(self.xedges)

        # helpers 1: MODIFIED: 接收 mutation_count 和 elapsed_time
        def record(input: np.ndarray, reward: float, oracle: bool, behavior: np.ndarray, final_state: np.ndarray, mutation_count: int, elapsed_time: float) -> None:
            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()
            # MODIFIED: 传递 mutation_count 和 elapsed_time
            updated_cell_index = self.update_cell(cell, input, reward, oracle, behavior, mutation_count, elapsed_time)
            # parent's cell is not logged
            print(f'episode_reward: {reward}, oracle: {float(oracle)}, cell_updated_index: {updated_cell_index}, nb_cells: {len(self.cells)}', file=logs_buffer)
            np.savetxt(inputs_buffer, input.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
            np.savetxt(final_states_buffer, final_state.reshape(1, -1), delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
        
        # helpers 2: MODIFIED: 接收和传递 mutation_counts, 计算时间
        def evaluate(individuals: np.ndarray, mutation_counts: np.ndarray, loop_start_time: float = None) -> np.ndarray:
            behaviors = []
            for i, ind in enumerate(individuals):
                # MODIFIED: 在执行前检查时间
                if (time.time() - testing_start_time > time_budget_seconds):
                    print("Time budget reached during evaluation. Stopping.")
                    break # 停止 evaluate 循环
                
                r, o, b, fs, _ = execute_policy(ind, model, env_seed, self.descriptors, 300)
                
                # MODIFIED: 计算时间
                if loop_start_time is None:
                    e_time = 0.0 # 初始化阶段
                else:
                    e_time = time.time() - loop_start_time

                # MODIFIED: 传递 mutation_counts[i] 和 e_time
                record(ind, r, o, b, fs, mutation_counts[i], e_time)
                behaviors.append(b)
            return np.array(behaviors)
        
        # helper 3: mutates a batch of individuals
        def mutate(inputs: np.ndarray):
            mutants = [self.mutate(input) for input in inputs]
            return np.array(mutants)

        # ns logs
        ns_logs_buffer = open(f'{filepath}_ns_logs.txt', 'w', buffering=1)
        nov_scores_buffer = open(f'{filepath}_nov_scores.txt', 'w', buffering=1)
        # initial population and novelty archive
        from novelty_search import NoveltyArchive
        print("Starting initial population evaluation...")
        pop = self.rng.integers(low=1, high=4, size=(pop_size, 15))
        # MODIFIED: 创建初始 counts
        pop_mutation_counts = np.zeros(pop_size, dtype=int)
        
        # MODIFIED: 传递 counts, loop_start_time=None
        pop_behaviors = evaluate(pop, pop_mutation_counts, loop_start_time=None)
        
        # 检查 evaluate 是否因时间耗尽而提前退出
        if not pop_behaviors.any():
             print("Time budget reached before initial population could be evaluated.")
             ns_logs_buffer.close()
             nov_scores_buffer.close()
             behaviors_buffer.close()
             inputs_buffer.close()
             cells_buffer.close()
             logs_buffer.close()
             final_states_buffer.close()
             return # 提前退出

        nov_archive = NoveltyArchive(pop_behaviors, k, nov_threshold)
        pop_nov_scores = nov_archive.score(pop_behaviors)
        [np.savetxt(nov_scores_buffer, s.reshape(1, -1), delimiter=',') for s in pop_nov_scores]
        # novelty search loop
        print(f'iteration: 0, archive_size: {nov_archive.size()}, archive_sparseness: {nov_archive.archive_sparseness():0.5f}', file=ns_logs_buffer)
        
        # MODIFIED: 更改为 while 循环
        i = 1
        pbar = tqdm.tqdm(disable=disable_pbar)
        pbar.set_description(f"Novelty Search loop (running for {time_budget_hours}h total)")

        print("Starting Novelty Search loop...")
        
        # MODIFIED: 记录主循环开始时间
        ns_start_time = time.time()

        while (time.time() - testing_start_time < time_budget_seconds):
            # 1. generates offspring
            offspring = mutate(pop)
            # MODIFIED: 创建后代的 counts
            offspring_mutation_counts = pop_mutation_counts + 1
            
            # 1. evaluates the offspring
            # MODIFIED: 传递 counts 和 start_time
            offspring_behaviors = evaluate(offspring, offspring_mutation_counts, loop_start_time=ns_start_time)
            
            # 检查 evaluate 是否耗尽了时间
            if not offspring_behaviors.any():
                print("Time budget reached during offspring evaluation. Breaking loop.")
                break

            # 1. novelty scores of the offspring w.r.t the archive and the population
            offspring_nov_scores = nov_archive.score(offspring_behaviors, pop_behaviors)

            # 2. selects the most novel individuals to form the new population
            joined_pop = np.vstack([pop, offspring])
            joined_scores = np.hstack([pop_nov_scores, offspring_nov_scores])
            # MODIFIED: 合并 counts
            joined_mutation_counts = np.hstack([pop_mutation_counts, offspring_mutation_counts])
            
            median_score = np.median(joined_scores)

            # 3. updates the archive
            _updated, _offspring_indices = nov_archive.update3(offspring_behaviors)

            # 4. updates the population and their data
            mask = (joined_scores >= median_score)

            pop = joined_pop[mask].copy()
            pop_behaviors = np.vstack([pop_behaviors, offspring_behaviors])[mask]
            # MODIFIED: 筛选 counts
            pop_mutation_counts = joined_mutation_counts[mask]
            
            pop_nov_scores = nov_archive.score(pop_behaviors)
            if len(pop) > pop_size:
                pop = pop[:pop_size]
                pop_behaviors = pop_behaviors[:pop_size]
                pop_nov_scores = pop_nov_scores[:pop_size]
                # MODIFIED: 截断 counts
                pop_mutation_counts = pop_mutation_counts[:pop_size]

            # (asserts ...)
            [np.savetxt(nov_scores_buffer, s.reshape(1, -1), delimiter=',') for s in pop_nov_scores]
            print(f'iteration: {i}, archive_size: {nov_archive.size()}, archive_sparseness: {nov_archive.archive_sparseness():0.5f}', file=ns_logs_buffer)
            
            i += 1
            pbar.update(1)

        print("Time budget reached. Stopping novelty_search.")
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
        MODIFIED: Returns both input and its mutation_count.
        '''
        scores = list(map(lambda x: x[1], self.cells_data[index]))
        # the best performing input is one whose score is the minimum, since it corresponds to the accumulated reward.
        best_performer_index = int(np.argmin(scores))
        
        # MODIFIED: 获取完整的元组
        # 结构: (input, score, is_faulty, behavior, mutation_count, elapsed_time)
        selected_data = self.cells_data[index][best_performer_index]
        
        # 返回 input (索引 0) 和 mutation_count (索引 4)
        return selected_data[0], selected_data[4]


if __name__ == '__main__':
    torch.set_num_threads(1)
    main_seed = 2021
    env_seed = 0
    model = load_model()

    # experimental parameters
    # MODIFIED: 定义时间预算
    time_budget_hours = 12
    
    # MODIFIED: test_policy 仍需要 init_budget
    init_budget = 1000
    test_budget=5000
    cell_granularity = 50

    # MODIFIED: novelty_search 仍需要这些参数
    population_size = 100
    k = 3
    novelty_threshold = 0.005

    results_fp = 'results/bw'
    if not os.path.exists(results_fp):
        os.makedirs(results_fp)

    for seed in EXPERIMENT_SEEDS:
        print(f'Seed {seed} starts.')
        for expert_indices in EXPERT_INDICES:
            print(f"--- Running MAP-Elites for {time_budget_hours} hours ---")
            f = MAPElitesFramework(seed, cell_granularity, descriptors=expert_indices, name='MAP-Elites')
            # MODIFIED: 更新了方法调用
            f.test_policy(model, env_seed, test_budget, init_budget, results_fp)
            
            #print(f"--- Running Novelty Search for {time_budget_hours} hours ---")
            #f = Framework(seed, cell_granularity, descriptors=expert_indices, name=f'Novelty Search')
            # MODIFIED: 更新了方法调用
            #f.novelty_search(
                #model, env_seed,
                #population_size,
                #k,
                #novelty_threshold,
                #time_budget_hours,
                #results_fp
            #)
        print(f'Experts done.')