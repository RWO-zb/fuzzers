import json
import os
import time
import tqdm
import numpy as np
import pandas as pd
from typing import List, Optional
from mc_utils import execute_policy, get_edges, compute_cell

class MAPElitesFramework:
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        self.rng = np.random.default_rng(rand_seed)
        self.creation_time = time.time()
        self.granularity = cell_granularity
        self.descriptors = np.array(descriptors)
        
        self.cells = []
        # 数据结构修改: (input, performance, oracle, behavior, mutation_count, discovery_time, seed_id)
        # 增加了最后一项 seed_id 用于溯源
        self.cells_data = []

        self.config = {
            'rand_seed': rand_seed,
            'cell_granularity': cell_granularity,
            'name': kwargs.get('name', 'MC-MAP-Elites')
        }

    def save_state(self, filepath: str):
        cell_dfs = []
        base_cols = ['score', 'is_faulty', 'cell_index'] + [f'cell{i}' for i in range(2)] + [f'behavior{i}' for i in range(2)]
        # [修改] CSV 列头增加 'seed_id'
        all_cols = base_cols + ['input', 'mutation_count', 'discovery_time', 'seed_id']

        for i, cell_data in enumerate(self.cells_data):
            # [修改] 解包时增加 seed_id，并在构建列表时包含它
            cell_dfs.append(pd.DataFrame.from_records(
                data=[[score, is_faulty, i] + self.cells[i] + behavior.tolist() + [inp.tolist()] + [cnt] + [dt] + [sid]
                      for (inp, score, is_faulty, behavior, cnt, dt, sid) in cell_data],
                columns=all_cols
            ))
        
        if cell_dfs:
            pd.concat(cell_dfs, ignore_index=True).to_csv(f'{filepath}_data.csv', index=False)
            
        with open(f'{filepath}_config.json', 'w') as f:
            json.dump(self.config, f)

    def select_input(self, index: int):
        scores = [x[1] for x in self.cells_data[index]]
        best_idx = int(np.argmin(scores))
        selected = self.cells_data[index][best_idx]
        # [修改] 返回值增加 seed_id (selected[6])
        return selected[0], selected[4], selected[6]

    def select_cell(self):
        return int(self.rng.integers(0, len(self.cells)))

    # [修改] 增加 seed_id 参数
    def update_cell(self, cell, inp, perf, is_faulty, beh, cnt, discovery_time, seed_id):
        try:
            idx = self.cells.index(cell)
            # [修改] 存储时包含 seed_id
            self.cells_data[idx].append((inp, perf, is_faulty, beh, cnt, discovery_time, seed_id))
        except ValueError:
            self.cells.append(cell)
            # [修改] 存储时包含 seed_id
            self.cells_data.append([(inp, perf, is_faulty, beh, cnt, discovery_time, seed_id)])
            idx = len(self.cells) - 1
        return idx

    def mutate(self, input_vec):
        noise = self.rng.normal(0, 0.05, size=input_vec.shape)
        mutated = input_vec + noise
        mutated[0] = np.clip(mutated[0], -0.6, -0.4)
        mutated[1] = 0.0 
        return mutated.astype(np.float32)

    def generate_random_input(self):
        pos = self.rng.uniform(-0.6, -0.4)
        return np.array([pos, 0.0], dtype=np.float32)

    def save_trajectory_log(self, file_handle, generation, input_vec, is_faulty, trajectory):
        header_info = {
            "Generation": int(generation),
            "Input": input_vec.tolist(),
            "Oracle": bool(is_faulty),
            "Steps": len(trajectory)
        }
        file_handle.write(f"--- Test Case Info: {json.dumps(header_info)} ---\n")
        for obs in trajectory:
            file_handle.write(f"{obs[0]:.6f}, {obs[1]:.6f}\n")

    def test_policy(self, model, env_seed, results_fp, init_budget=1000, time_budget_hours=None, max_samples=None):
        """
        执行测试策略。
        
        参数:
        - init_budget: 初始化阶段尝试生成的随机样本数。
        - time_budget_hours: 时间预算（小时）。如果为 None，则不限制时间。
        - max_samples: 最大测试样例数（Total Budget）。如果为 None，则不限制数量。
        注意: time_budget_hours 和 max_samples 至少需要提供一个，否则程序将无限运行（除非手动停止）。
        """
        
        # --- 1. 预算参数处理 ---
        start_time = time.time()
        
        # 将时间转换为秒，如果为None则设为无穷大
        time_limit_sec = (time_budget_hours * 3600) if time_budget_hours is not None else float('inf')
        
        # 样本数限制，如果为None则设为无穷大
        sample_limit = max_samples if max_samples is not None else float('inf')

        # 计数器：记录已执行的测试用例总数 (execute_policy 调用的次数)
        total_executions = 0

        # 辅助函数：检查是否耗尽预算
        def is_budget_exhausted():
            time_used = time.time() - start_time
            if time_used >= time_limit_sec:
                return True, "Time Budget Exceeded"
            if total_executions >= sample_limit:
                return True, "Sample Budget Exceeded"
            return False, ""

        # --- 2. 文件准备 ---
        if os.path.isdir(results_fp):
            filepath = os.path.join(results_fp, str(start_time))
        else:
            filepath = results_fp

        files = {
            'inputs': open(f'{filepath}_inputs.txt', 'w'),
            'behaviors': open(f'{filepath}_behaviors.txt', 'w'),
            'cells': open(f'{filepath}_cells.txt', 'w'),
            'logs': open(f'{filepath}_logs.txt', 'w'),
            'final_states': open(f'{filepath}_final_states.txt', 'w'),
            'obs': open(f'{filepath}_obs.txt', 'w')
        }

        print(f"Starting Initialization (Target: {init_budget} samples)...")
        print(f"Constraints -> Time: {time_budget_hours}h, Max Samples: {max_samples}")

        inputs, behaviors, acc_rewards, oracles = [], [], [], []
        discovery_times = [] 
        
        # --- 3. Initialization Phase ---
        # 进度条显示初始化的进度
        pbar_init = tqdm.tqdm(total=init_budget, desc="Init Phase")
        
        while len(inputs) < init_budget:
            # 检查总预算
            exhausted, reason = is_budget_exhausted()
            if exhausted:
                print(f"\n[Stopping] Initialization stopped: {reason}")
                break
            
            inp = self.generate_random_input()
            
            # 执行策略 (消耗 1 个 budget)
            rew, oracle, beh, fs, traj, _ = execute_policy(inp, model, env_seed)
            total_executions += 1
            
            dt = time.time() - start_time
            discovery_times.append(dt)

            inputs.append(inp)
            behaviors.append(beh)
            acc_rewards.append(rew)
            oracles.append(oracle)
            
            self.save_trajectory_log(files['obs'], 0, inp, oracle, traj)
            pbar_init.update(1)
            
        pbar_init.close()

        # 如果初始化期间就耗尽了预算，直接保存退出
        exhausted, _ = is_budget_exhausted()
        if exhausted and len(inputs) < init_budget:
             print("Budget exhausted during initialization. Skipping fuzzing loop.")
             # 处理已有的数据以便保存
             if len(inputs) > 0:
                behaviors = np.array(behaviors)
                self.xedges, self.yedges = get_edges(env_seed, self.descriptors)
                for i in range(len(inputs)):
                    cell = compute_cell(behaviors[i], self.xedges, self.yedges).tolist()
                    # [修改] 传入 seed_id=i，将初始化的索引作为种子ID
                    self.update_cell(cell, inputs[i], acc_rewards[i], oracles[i], behaviors[i], 0, discovery_times[i], seed_id=i)
             for f in files.values(): f.close()
             self.save_state(filepath)
             return

        # 处理初始化数据建立 Map
        behaviors = np.array(behaviors)
        self.xedges, self.yedges = get_edges(env_seed, self.descriptors)
        
        for i in range(len(inputs)):
            cell = compute_cell(behaviors[i], self.xedges, self.yedges).tolist()
            # [修改] 传入 seed_id=i
            self.update_cell(cell, inputs[i], acc_rewards[i], oracles[i], behaviors[i], 0, discovery_times[i], seed_id=i)
            
            np.savetxt(files['inputs'], inputs[i].reshape(1, -1), fmt='%f', delimiter=',')
            np.savetxt(files['behaviors'], behaviors[i].reshape(1, -1), delimiter=',')
            np.savetxt(files['cells'], np.array(cell).reshape(1, -1), fmt='%d', delimiter=',')

        print("Starting Fuzzing Loop...")
        pbar = tqdm.tqdm(desc="Fuzzing Phase")
        
        # --- 4. Fuzzing Loop ---
        while True:
            # 检查总预算
            exhausted, reason = is_budget_exhausted()
            if exhausted:
                print(f"\n[Stopping] Fuzzing stopped: {reason}")
                break

            cell_idx = self.select_cell()
            # [修改] 接收返回的 parent_seed_id
            parent_inp, parent_cnt, parent_seed_id = self.select_input(cell_idx)
            
            mutated_inp = self.mutate(parent_inp)
            
            # 执行策略 (消耗 1 个 budget)
            rew, oracle, beh, fs, traj, _ = execute_policy(mutated_inp, model, env_seed)
            total_executions += 1
            
            current_discovery_time = time.time() - start_time
            current_generation = parent_cnt + 1

            cell = compute_cell(beh, self.xedges, self.yedges).tolist()
            
            # [修改] 传入继承的 parent_seed_id
            self.update_cell(cell, mutated_inp, rew, oracle, beh, current_generation, current_discovery_time, seed_id=parent_seed_id)
            
            self.save_trajectory_log(files['obs'], current_generation, mutated_inp, oracle, traj)

            np.savetxt(files['inputs'], mutated_inp.reshape(1, -1), fmt='%f', delimiter=',')
            np.savetxt(files['behaviors'], beh.reshape(1, -1), delimiter=',')
            np.savetxt(files['cells'], np.array(cell).reshape(1, -1), fmt='%d', delimiter=',')
            
            pbar.update(1)
            # 可选：更新进度条后缀显示当前状态
            # [注意] 这里原代码的列表推导式需要解包7个值，或者改用索引。
            # 原代码: np.sum([1 for c in self.cells_data for x in c if x[2]])
            # 这里的 x[2] 对应 is_faulty，依然有效，不需要修改
            pbar.set_postfix({'Execs': total_executions, 'Crashes': np.sum([1 for c in self.cells_data for x in c if x[2]])})
            
        pbar.close()
        for f in files.values(): f.close()
        self.save_state(filepath)
        print(f"Experiment Completed. Total Executions: {total_executions}")