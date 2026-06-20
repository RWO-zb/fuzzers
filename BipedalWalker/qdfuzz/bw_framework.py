import json
import argparse
import os
import time
import torch
import tqdm
import numpy as np
import pandas as pd
import random 
import pickle 

from stable_baselines3.common.base_class import BaseAlgorithm
from typing import List, Optional

from bw_common import load_model, EXPERT_INDICES, execute_policy, get_edges
from common import compute_cell, EXPERIMENT_SEEDS

# ==========================================
# [辅助函数] TodyNet 数据收集 (保持不变)
# ==========================================
def process_episode_data(sequence, label, window_size):
    seq_len = len(sequence)
    if seq_len < window_size:
        return None, None
    
    seq_array = np.array(sequence) 
    windows = []
    labels = []

    if label == 0:
        max_idx = seq_len - window_size
        rand_idx = random.randint(0, max_idx)
        win = seq_array[rand_idx : rand_idx + window_size]
        win = win.transpose() 
        windows.append(win)
        labels.append(0)
    else:
        win = seq_array[-window_size:] 
        win = win.transpose()
        windows.append(win)
        labels.append(1)
        
    return np.array(windows), np.array(labels)

def balance_and_save_data(X_list, y_list, output_dir, dataset_name, window_size, target_total=3000, target_crash_ratio=0.30):
    if not X_list:
        return
    
    print(f"\n[TodyNet Data] Processing balancing to {target_total} samples (Target Crash Ratio: {target_crash_ratio:.0%})...")
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    indices_fail = np.where(y_all == 1)[0]
    indices_succ = np.where(y_all == 0)[0]
    
    n_crash_target = int(target_total * target_crash_ratio)
    n_succ_target = target_total - n_crash_target
    
    print(f"  Raw Collected: Fail={len(indices_fail)}, Success={len(indices_succ)}")

    if len(indices_fail) >= n_crash_target:
        final_fail = np.random.choice(indices_fail, size=n_crash_target, replace=False)
    else:
        final_fail = indices_fail
        
    if len(indices_succ) >= n_succ_target:
        final_succ = np.random.choice(indices_succ, size=n_succ_target, replace=False)
    else:
        final_succ = indices_succ
    
    final_indices = np.concatenate([final_fail, final_succ])
    np.random.shuffle(final_indices)
    
    X_balanced = X_all[final_indices]
    y_balanced = y_all[final_indices]

    X_final = np.expand_dims(X_balanced, axis=1)
    X_tensor = torch.from_numpy(X_final).float()
    y_tensor = torch.from_numpy(y_balanced).long()
    
    total = X_tensor.size(0)
    indices = torch.randperm(total)
    split = int(0.8 * total)
    
    ds_id = f"{dataset_name}_{window_size}"
    save_path = os.path.join(output_dir, ds_id)
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(X_tensor[indices[:split]], os.path.join(save_path, 'X_train.pt'))
    torch.save(y_tensor[indices[:split]], os.path.join(save_path, 'y_train.pt'))
    torch.save(X_tensor[indices[split:]], os.path.join(save_path, 'X_valid.pt'))
    torch.save(y_tensor[indices[split:]], os.path.join(save_path, 'y_valid.pt'))
    
    print(f"[TodyNet Data] Saved {total} samples to {save_path}")

# ==========================================

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

        self.last_cell_selected = None
        self.last_cell_updated = None

        self.cells: list[list[int]] = []
        self.cells_data: list[list[tuple[np.ndarray, float, bool, np.ndarray, int, float, int]]] = []

        self.config = {
            'rand_seed': self.rand_seed,
            'cell_granularity': self.granularity,
            'descriptors': self.descriptors.tolist(),
            'use_case': 'Bipedal Walker'
        }

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
        if not filepath.endswith('config'):
            filepath += '_config'
        f = open(f'{filepath}.json', 'w')
        f.write(json.dumps(self.config))
        f.close()

    def save_random_state(self, filepath: str):
        f = open(f'{filepath}_state.json', 'w')
        f.write(json.dumps(self.rng.bit_generator.state))
        f.close()
        return self.rng.bit_generator.state

    def save_state(self, filepath: str):
        cell_dfs = []
        base_columns = ['score', 'is_faulty', 'cell_index'] + [f'cell{i}' for i in range(2)] + [f'behavior{i}' for i in range(2)]
        all_columns = base_columns + ['input', 'mutation_count', 'elapsed_time', 'seed_id']

        for i, cell_data in enumerate(self.cells_data):
            records = []
            for item in cell_data:
                _input, score, is_faulty, behavior, mutation_count, elapsed_time, seed_id = item
                record = [
                    score, 
                    is_faulty, 
                    i, 
                    self.cells[i][0], self.cells[i][1],
                    behavior[0], behavior[1],
                    json.dumps(_input.tolist()),
                    mutation_count,
                    elapsed_time,
                    seed_id
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
        if not filepath.endswith('config'):
            filepath += '_config'
        f = open(f'{filepath}.json', 'r')
        self.config = json.load(f)
        f.close()

    def load_random_state(self, filepath: str):
        if not filepath.endswith('state'):
            filepath += '_state'
        f = open(f'{filepath}.json', 'r')
        self.rng.bit_generator.state = json.load(f)
        f.close()

    def load_state(self, filepath: str):
        df_fp = f'{filepath}_data.csv'
        assert os.path.exists(df_fp), 'file is missing.'
        self.cells = []
        self.cells_data = []

        df = pd.read_csv(df_fp)

        cell_cols = [c for c in df.columns.to_list() if c.startswith('cell') and 'index' not in c]
        behavior_cols = [c for c in df.columns.to_list() if c.startswith('behavior')]

        has_elapsed_time = 'elapsed_time' in df.columns
        has_mutation_count = 'mutation_count' in df.columns
        has_seed_id = 'seed_id' in df.columns

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
            seed_id = int(row['seed_id']) if has_seed_id else -1
            
            self.update_cell(cell, input_vec, performance, is_faulty, np.array(behavior), mutation_count, elapsed_time, seed_id)

        self.load_random_state(filepath)
        self.load_configuration(filepath)
        self.loaded = True
        return len(df)

    def select_input(self, index: int):
        input_index: int = self.rng.integers(0, len(self.cells_data[index]))
        selected_data = self.cells_data[index][input_index]
        return selected_data[0], selected_data[4], selected_data[6]

    def select_cell(self):
        return int(self.rng.integers(0, len(self.cells)))

    def update_cell(self, cell: List[int], input: np.ndarray, performance: float, is_faulty: bool, behavior: np.ndarray, mutation_count: int, elapsed_time: float, seed_id: int):
        index = None
        try:
            index = self.cells.index(cell)
            self.cells_data[index].append((input, performance, is_faulty, behavior, mutation_count, elapsed_time, seed_id))
        except ValueError:
            self.cells.append(cell)
            self.cells_data.append([(input, performance, is_faulty, behavior, mutation_count, elapsed_time, seed_id)])
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
                    time_budget_hours: Optional[float] = None,
                    execution_budget: Optional[int] = None,
                    disable_pbar: bool = False,
                    save_data: bool = True,
                    window_size: int = 25
                    ):

        if time_budget_hours is None and execution_budget is None:
            raise ValueError("At least one budget must be provided.")

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

        print(f'Starting test_policy. Time Budget: {time_budget_hours}h, Execution Budget: {execution_budget} (Init not counted)')

        inputs: List[np.ndarray] = []
        behaviors = []
        final_states: List[np.ndarray] = []
        acc_rewards: List[float] = []
        oracles: List[bool] = []
        seed_ids: List[int] = [] 
        
        testing_start_time = time.time() 
        execution_times = []
        n_executions = 0 
        
        # [容器]
        all_window_data = [] 
        all_label_data = []
        todynet_success_count = 0
        TODYNET_SUCCESS_CAP = 3000

        crash_transitions = []
        success_transitions = []
        TRANSITION_CRASH_CAP = 10000
        TRANSITION_SUCCESS_CAP = 90000

        # [新增] 对齐评估指标：预留列表和累计计时
        eval_selection_log = []
        total_env_sim_time = 0.0

        print("Starting initialization phase...")
        for i in tqdm.tqdm(range(init_budget), disable=disable_pbar):
            input: np.ndarray = self.rng.integers(low=1, high=4, size=15)
            
            current_seed_id = i 

            t0 = time.time()
            # [修改] 提取 8 个返回值，包括 eval_info
            episode_reward, oracle, behavior, fs, _, _, _, _ = execute_policy(input, model, env_seed, self.descriptors)
            t1 = time.time()
            execution_times.append(t1 - t0)

            inputs.append(input)
            behaviors.append(behavior)
            final_states.append(fs)
            acc_rewards.append(episode_reward)
            oracles.append(oracle)
            seed_ids.append(current_seed_id) 
            n_executions += 1
        
        if not inputs:
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
            mutated_input_index = self.update_cell(cell, inputs[i], acc_rewards[i], oracles[i], behavior, 0, 0.0, seed_ids[i])
            print(f'episode_reward: {acc_rewards[i]}, oracle: {float(oracles[i])}, cell_selected_index: -1, cell_updated_index: {mutated_input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, inputs[i].reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
            np.savetxt(final_states_buffer, final_states[i].reshape(1, -1), delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')

        fuzz_executions = 0
        fuzzing_start_time = time.time()

        if execution_budget is not None:
            pbar = tqdm.tqdm(total=execution_budget, initial=0, disable=disable_pbar)
            pbar.set_description(f"Fuzzing (max {execution_budget} execs)")
        else:
            pbar = tqdm.tqdm(disable=disable_pbar)
            pbar.set_description(f"Fuzzing (max {time_budget_hours}h)")

        print("Starting fuzzing loop (Data Collection Active)...")
        
        while self._check_budget(fuzzing_start_time, fuzz_executions, time_budget_hours, execution_budget):
            cell_index = self.select_cell()
            self.last_cell_selected = cell_index
            input, parent_mutation_count, parent_seed_id = self.select_input(cell_index)

            mutated_input = self.mutate(input)
            t0 = time.time()
            
            # [修改] 提取 eval_info，以支持统一评估指标的数据收集
            episode_reward, oracle, behavior, fs, _, todynet_trace, rl_data, eval_info = execute_policy(mutated_input, model, env_seed, self.descriptors)
            t1 = time.time()
            execution_times.append(t1 - t0)
            
            n_executions += 1 
            fuzz_executions += 1 

            # [新增] 对齐评估指标：累加时间开销和收集日志
            total_env_sim_time += eval_info['env_sim_time']
            eval_selection_log.append({
                'mutate_state': mutated_input.copy(),
                'did_crash': eval_info['did_crash'],
                'is_reward_fault': eval_info['is_reward_fault'],
                'env_id': eval_info['env_id'],
                'env_seed': eval_info['env_seed'],
                'sim_steps': eval_info['sim_steps'],
                'elapsed_time': time.time() - fuzzing_start_time,
                'survival_steps': eval_info['survival_steps'],
                'parent_depth': parent_mutation_count,
                'output_trajectory': eval_info['output_trajectory']
            })

            # [Fuzz阶段] 数据收集逻辑
            if save_data:
                is_crash = oracle
                label = 1 if is_crash else 0
                
                # TodyNet
                collect_todynet = True if is_crash else (todynet_success_count < TODYNET_SUCCESS_CAP)
                if collect_todynet:
                    wins, labels = process_episode_data(todynet_trace, label, window_size)
                    if wins is not None:
                        all_window_data.append(wins)
                        all_label_data.append(labels)
                        if not is_crash:
                            todynet_success_count += 1
                
                # RL Transitions
                if is_crash:
                    if len(crash_transitions) < TRANSITION_CRASH_CAP:
                        crash_transitions.extend(rl_data)
                else:
                    if len(success_transitions) < TRANSITION_SUCCESS_CAP:
                        success_transitions.extend(rl_data)

            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()
            new_mutation_count = parent_mutation_count + 1
            elapsed_time = time.time() - fuzzing_start_time
            
            mutated_input_index = self.update_cell(cell, mutated_input, episode_reward, oracle, behavior, new_mutation_count, elapsed_time, parent_seed_id)
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
        
        if save_data:
            balance_and_save_data(all_window_data, all_label_data, results_fp, "BipedalWalkerHC", window_size)
            
            trans_file = os.path.join(results_fp, 'transitions.pkl')
            print(f"Saving RL transitions to {trans_file}...")
            save_dict = {
                "crash": crash_transitions,
                "success": success_transitions,
                "is_raw": True
            }
            with open(trans_file, 'wb') as f_t:
                pickle.dump(save_dict, f_t, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"RL Transitions saved. (Crash: {len(crash_transitions)}, Success: {len(success_transitions)})")

        # [新增] 保存对齐评估指标与开销
        out_dir = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
        with open(os.path.join(out_dir, 'selection_log.pkl'), 'wb') as f:
            pickle.dump(eval_selection_log, f)
            
        perf_meta = {
            'total_wall_time': time.time() - fuzzing_start_time,
            'env_sim_time': total_env_sim_time,
            'algo_logic_time': (time.time() - fuzzing_start_time) - total_env_sim_time
        }
        with open(os.path.join(out_dir, 'perf_meta.pkl'), 'wb') as f:
            pickle.dump(perf_meta, f)


    def random_testing(self, model: BaseAlgorithm,
                    env_seed: int,
                    results_fp: str,
                    time_budget_hours: Optional[float] = None,
                    execution_budget: Optional[int] = None,
                    disable_pbar: bool = False,
                    save_data: bool = True,
                    window_size: int = 25
                    ):
        '''Random testing loop baseline. (All phases considered fuzzing/testing)'''
        
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
        
        # [容器]
        all_window_data = [] 
        all_label_data = []
        todynet_success_count = 0
        TODYNET_SUCCESS_CAP = 3000

        crash_transitions = []
        success_transitions = []
        TRANSITION_CRASH_CAP = 10000
        TRANSITION_SUCCESS_CAP = 90000

        # [新增] 对齐评估指标：预留列表和累计计时
        eval_selection_log = []
        total_env_sim_time = 0.0
        
        if execution_budget is not None:
            pbar = tqdm.tqdm(total=execution_budget, disable=disable_pbar)
        else:
            pbar = tqdm.tqdm(disable=disable_pbar)

        while self._check_budget(start_time, n_executions, time_budget_hours, execution_budget):
            input: np.ndarray = self.rng.integers(low=1, high=4, size=15)
            current_seed_id = n_executions 
            
            t0 = time.time()
            
            # [修改] 解包增加 eval_info
            episode_reward, oracle, behavior, fs, _, todynet_trace, rl_data, eval_info = execute_policy(input, model, env_seed, self.descriptors)
            t1 = time.time()
            execution_times.append(t1 - t0)
            
            n_executions += 1 

            # [新增] 对齐评估指标收集
            total_env_sim_time += eval_info['env_sim_time']
            eval_selection_log.append({
                'mutate_state': input.copy(),
                'did_crash': eval_info['did_crash'],
                'is_reward_fault': eval_info['is_reward_fault'],
                'env_id': eval_info['env_id'],
                'env_seed': eval_info['env_seed'],
                'sim_steps': eval_info['sim_steps'],
                'elapsed_time': time.time() - start_time,
                'survival_steps': eval_info['survival_steps'],
                'parent_depth': 0,
                'output_trajectory': eval_info['output_trajectory']
            })

            if save_data:
                is_crash = oracle
                label = 1 if is_crash else 0
                
                collect_todynet = True if is_crash else (todynet_success_count < TODYNET_SUCCESS_CAP)
                if collect_todynet:
                    wins, labels = process_episode_data(todynet_trace, label, window_size)
                    if wins is not None:
                        all_window_data.append(wins)
                        all_label_data.append(labels)
                        if not is_crash:
                            todynet_success_count += 1
                
                if is_crash:
                    if len(crash_transitions) < TRANSITION_CRASH_CAP:
                        crash_transitions.extend(rl_data)
                else:
                    if len(success_transitions) < TRANSITION_SUCCESS_CAP:
                        success_transitions.extend(rl_data)

            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()
            elapsed_time = time.time() - start_time

            input_index = self.update_cell(cell, input, episode_reward, oracle, behavior, 0, elapsed_time, current_seed_id)
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
        
        if save_data:
            balance_and_save_data(all_window_data, all_label_data, results_fp, "BipedalWalkerHC", window_size)
            
            trans_file = os.path.join(results_fp, 'transitions.pkl')
            print(f"Saving RL transitions to {trans_file}...")
            save_dict = {
                "crash": crash_transitions,
                "success": success_transitions,
                "is_raw": True
            }
            with open(trans_file, 'wb') as f_t:
                pickle.dump(save_dict, f_t, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"RL Transitions saved. (Crash: {len(crash_transitions)}, Success: {len(success_transitions)})")

        # [新增] 保存对齐评估指标与开销
        out_dir = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
        with open(os.path.join(out_dir, 'selection_log.pkl'), 'wb') as f:
            pickle.dump(eval_selection_log, f)
            
        perf_meta = {
            'total_wall_time': time.time() - start_time,
            'env_sim_time': total_env_sim_time,
            'algo_logic_time': (time.time() - start_time) - total_env_sim_time
        }
        with open(os.path.join(out_dir, 'perf_meta.pkl'), 'wb') as f:
            pickle.dump(perf_meta, f)


    def novelty_search(self, model: BaseAlgorithm,
                    env_seed: int,
                    pop_size: int,
                    k: int,
                    nov_threshold: float,
                    results_fp: str,
                    time_budget_hours: Optional[float] = None,
                    execution_budget: Optional[int] = None,
                    disable_pbar: bool = False,
                    save_data: bool = True,
                    window_size: int = 25
                    ):
        '''Does not use cached data anymore. Budget only counts evolutionary phase.'''

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
        n_executions = 0 # 总执行次数
        fuzz_executions = 0 # 仅 Fuzz 阶段执行次数
        
        # [容器]
        all_window_data = [] 
        all_label_data = []
        todynet_success_count = 0
        TODYNET_SUCCESS_CAP = 3000

        crash_transitions = []
        success_transitions = []
        TRANSITION_CRASH_CAP = 10000
        TRANSITION_SUCCESS_CAP = 90000

        # [新增] 对齐评估指标：预留列表和累计计时
        eval_selection_log = []
        total_env_sim_time = 0.0
        
        print(f'Starting novelty_search. Time Budget: {time_budget_hours}h, Execution Budget: {execution_budget} (Init not counted)')

        self.xedges, self.yedges = get_edges(env_seed, self.descriptors)
        self.config['xedges'] = list(self.xedges)
        self.config['yedges'] = list(self.xedges)

        def record(input: np.ndarray, reward: float, oracle: bool, behavior: np.ndarray, final_state: np.ndarray, mutation_count: int, elapsed_time: float, seed_id: int) -> None:
            cell = compute_cell(behavior, self.xedges, self.yedges).tolist()
            updated_cell_index = self.update_cell(cell, input, reward, oracle, behavior, mutation_count, elapsed_time, seed_id)
            print(f'episode_reward: {reward}, oracle: {float(oracle)}, cell_updated_index: {updated_cell_index}, nb_cells: {len(self.cells)}', file=logs_buffer)
            np.savetxt(inputs_buffer, input.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
            np.savetxt(final_states_buffer, final_state.reshape(1, -1), delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
        
        def evaluate(individuals: np.ndarray, mutation_counts: np.ndarray, seed_ids: np.ndarray, loop_start_time: float = None, check_budget: bool = True, collect_data: bool = True) -> np.ndarray:
            nonlocal n_executions 
            nonlocal fuzz_executions
            nonlocal todynet_success_count 
            nonlocal total_env_sim_time # [新增] 对齐评估指标需要
            
            behaviors = []
            for i, ind in enumerate(individuals):
                if check_budget:
                    if not self._check_budget(loop_start_time, fuzz_executions, time_budget_hours, execution_budget):
                        break 
                
                # [修改] 提取 eval_info
                episode_reward, oracle, behavior, fs, _, todynet_trace, rl_data, eval_info = execute_policy(ind, model, env_seed, self.descriptors, 300)
                n_executions += 1
                
                if check_budget:
                    fuzz_executions += 1 
                    
                    # [新增] 仅在Fuzzing阶段进行对齐评估指标的收集
                    total_env_sim_time += eval_info['env_sim_time']
                    eval_selection_log.append({
                        'mutate_state': ind.copy(),
                        'did_crash': eval_info['did_crash'],
                        'is_reward_fault': eval_info['is_reward_fault'],
                        'env_id': eval_info['env_id'],
                        'env_seed': eval_info['env_seed'],
                        'sim_steps': eval_info['sim_steps'],
                        'elapsed_time': time.time() - loop_start_time,
                        'survival_steps': eval_info['survival_steps'],
                        'parent_depth': mutation_counts[i],
                        'output_trajectory': eval_info['output_trajectory']
                    })
                
                if collect_data and save_data:
                    is_crash = oracle
                    label = 1 if is_crash else 0
                    
                    collect_todynet = True if is_crash else (todynet_success_count < TODYNET_SUCCESS_CAP)
                    if collect_todynet:
                        wins, labels = process_episode_data(todynet_trace, label, window_size)
                        if wins is not None:
                            all_window_data.append(wins)
                            all_label_data.append(labels)
                            if not is_crash:
                                todynet_success_count += 1
                    
                    if is_crash:
                        if len(crash_transitions) < TRANSITION_CRASH_CAP:
                            crash_transitions.extend(rl_data)
                    else:
                        if len(success_transitions) < TRANSITION_SUCCESS_CAP:
                            success_transitions.extend(rl_data)

                if loop_start_time is None:
                    e_time = 0.0
                else:
                    e_time = time.time() - loop_start_time

                record(ind, episode_reward, oracle, behavior, fs, mutation_counts[i], e_time, seed_ids[i])
                behaviors.append(behavior)
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
        
        pop_seed_ids = np.arange(pop_size, dtype=int)
        
        pop_behaviors = evaluate(pop, pop_mutation_counts, pop_seed_ids, loop_start_time=None, check_budget=False, collect_data=False)
        
        if not pop_behaviors.any():
             print("No behaviors generated in init.")
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
             pbar = tqdm.tqdm(total=execution_budget, initial=0, disable=disable_pbar)
        else:
             pbar = tqdm.tqdm(disable=disable_pbar)

        print("Starting Novelty Search loop (Data Collection Active)...")
        ns_start_time = time.time() 

        while self._check_budget(ns_start_time, fuzz_executions, time_budget_hours, execution_budget):
            offspring = mutate(pop)
            offspring_mutation_counts = pop_mutation_counts + 1
            offspring_seed_ids = pop_seed_ids.copy()
            
            prev_executions = fuzz_executions
            offspring_behaviors = evaluate(offspring, offspring_mutation_counts, offspring_seed_ids, loop_start_time=ns_start_time, check_budget=True, collect_data=True)
            
            executions_diff = fuzz_executions - prev_executions
            pbar.update(executions_diff)
            
            if not offspring_behaviors.any():
                print("Budget reached during offspring evaluation.")
                break

            offspring_nov_scores = nov_archive.score(offspring_behaviors, pop_behaviors)

            joined_pop = np.vstack([pop, offspring[:len(offspring_behaviors)]]) 
            joined_scores = np.hstack([pop_nov_scores, offspring_nov_scores])
            joined_mutation_counts = np.hstack([pop_mutation_counts, offspring_mutation_counts[:len(offspring_behaviors)]])
            joined_seed_ids = np.hstack([pop_seed_ids, offspring_seed_ids[:len(offspring_behaviors)]])
            
            median_score = np.median(joined_scores)

            _updated, _offspring_indices = nov_archive.update3(offspring_behaviors)

            mask = (joined_scores >= median_score)

            pop = joined_pop[mask].copy()
            pop_behaviors = np.vstack([pop_behaviors, offspring_behaviors])[mask]
            pop_mutation_counts = joined_mutation_counts[mask]
            pop_seed_ids = joined_seed_ids[mask]
            
            pop_nov_scores = nov_archive.score(pop_behaviors)
            if len(pop) > pop_size:
                pop = pop[:pop_size]
                pop_behaviors = pop_behaviors[:pop_size]
                pop_nov_scores = pop_nov_scores[:pop_size]
                pop_mutation_counts = pop_mutation_counts[:pop_size]
                pop_seed_ids = pop_seed_ids[:pop_size]

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
        
        if save_data:
            balance_and_save_data(all_window_data, all_label_data, results_fp, "BipedalWalkerHC", window_size)
            
            trans_file = os.path.join(results_fp, 'transitions.pkl')
            print(f"Saving RL transitions to {trans_file}...")
            save_dict = {
                "crash": crash_transitions,
                "success": success_transitions,
                "is_raw": True
            }
            with open(trans_file, 'wb') as f_t:
                pickle.dump(save_dict, f_t, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"RL Transitions saved. (Crash: {len(crash_transitions)}, Success: {len(success_transitions)})")

        # [新增] 保存对齐评估指标与开销
        out_dir = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
        with open(os.path.join(out_dir, 'selection_log.pkl'), 'wb') as f:
            pickle.dump(eval_selection_log, f)
            
        perf_meta = {
            'total_wall_time': time.time() - ns_start_time,
            'env_sim_time': total_env_sim_time,
            'algo_logic_time': (time.time() - ns_start_time) - total_env_sim_time
        }
        with open(os.path.join(out_dir, 'perf_meta.pkl'), 'wb') as f:
            pickle.dump(perf_meta, f)

class MAPElitesFramework(Framework):
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        if kwargs.get('name') is None:
            kwargs['name'] = 'MAP-Elites'
        super().__init__(rand_seed, cell_granularity, descriptors, **kwargs)


    def select_input(self, index: int):
        '''
        Returns input, mutation_count, and seed_id.
        '''
        scores = list(map(lambda x: x[1], self.cells_data[index]))
        best_performer_index = int(np.argmin(scores))
        selected_data = self.cells_data[index][best_performer_index]
        return selected_data[0], selected_data[4], selected_data[6]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["tqc", "ppo"], default="tqc")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vecnormalize-path", default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    main_seed = 1
    env_seed = 0
    model = load_model(
        algo=args.algo,
        model_path=args.model_path,
        vecnormalize_path=args.vecnormalize_path,
    )

    # --- 配置区域 ---
    TIME_BUDGET_HOURS = 12   
    EXECUTION_BUDGET = None   
    
    init_budget = 1000
    cell_granularity = 50

    population_size = 100
    k = 3
    novelty_threshold = 0.005

    run_tag = time.strftime("%Y%m%d-%H%M%S")
    results_root = os.path.join('results', 'bw', f'run_{run_tag}')

    for seed in EXPERIMENT_SEEDS:
        print(f'Seed {seed} starts.')
        for expert_indices in EXPERT_INDICES:
            print(f"--- Running MAP-Elites ---")
            desc_tag = 'desc_' + '_'.join(map(str, expert_indices))
            results_fp = os.path.join(results_root, f'seed_{seed}', desc_tag)
            os.makedirs(results_fp, exist_ok=True)

            f = MAPElitesFramework(seed, cell_granularity, descriptors=expert_indices, name='MAP-Elites')
            f.config['experiment_seed'] = seed
            f.config['results_dir'] = results_fp
            
            f.test_policy(
                model=model, 
                env_seed=env_seed, 
                init_budget=init_budget, 
                results_fp=results_fp,
                time_budget_hours=TIME_BUDGET_HOURS,    
                execution_budget=EXECUTION_BUDGET,
                save_data=False,
                window_size=25
            )
            
        print(f'Experts done.')
