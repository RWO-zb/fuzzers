import json
import os
import time
import tqdm
import numpy as np
import pandas as pd
from typing import List
from mc_utils import execute_policy, get_edges, compute_cell

class MAPElitesFramework:
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        self.rng = np.random.default_rng(rand_seed)
        self.creation_time = time.time()
        self.granularity = cell_granularity
        self.descriptors = np.array(descriptors)
        
        self.cells = []
        # 数据结构: (input, performance, oracle, behavior, mutation_count, discovery_time)
        self.cells_data = []

        self.config = {
            'rand_seed': rand_seed,
            'cell_granularity': cell_granularity,
            'name': kwargs.get('name', 'MC-MAP-Elites')
        }

    def save_state(self, filepath: str):
        cell_dfs = []
        base_cols = ['score', 'is_faulty', 'cell_index'] + [f'cell{i}' for i in range(2)] + [f'behavior{i}' for i in range(2)]
        all_cols = base_cols + ['input', 'mutation_count', 'discovery_time']

        for i, cell_data in enumerate(self.cells_data):
            cell_dfs.append(pd.DataFrame.from_records(
                data=[[score, is_faulty, i] + self.cells[i] + behavior.tolist() + [inp.tolist()] + [cnt] + [dt]
                      for (inp, score, is_faulty, behavior, cnt, dt) in cell_data],
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
        return selected[0], selected[4]

    def select_cell(self):
        return int(self.rng.integers(0, len(self.cells)))

    def update_cell(self, cell, inp, perf, is_faulty, beh, cnt, discovery_time):
        try:
            idx = self.cells.index(cell)
            self.cells_data[idx].append((inp, perf, is_faulty, beh, cnt, discovery_time))
        except ValueError:
            self.cells.append(cell)
            self.cells_data.append([(inp, perf, is_faulty, beh, cnt, discovery_time)])
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

    # [新增] 辅助函数：严格按照格式写入轨迹日志
    def save_trajectory_log(self, file_handle, generation, input_vec, is_faulty, trajectory):
        # 构建 Header 字典
        # JSON中的 bool 是 lowercase (true/false)，Python 是 True/False。json.dumps 会自动处理。
        header_info = {
            "Generation": int(generation),
            "Input": input_vec.tolist(),
            "Oracle": bool(is_faulty), # is_faulty 为 True 表示 Crash
            "Steps": len(trajectory)
        }
        
        # 写入 Header 行
        file_handle.write(f"--- Test Case Info: {json.dumps(header_info)} ---\n")
        
        # 写入数据行 (Pos, Vel)
        for obs in trajectory:
            file_handle.write(f"{obs[0]:.6f}, {obs[1]:.6f}\n")

    def test_policy(self, model, env_seed, time_budget_hours, init_budget, results_fp):
        time_budget_sec = time_budget_hours * 3600
        start_time = time.time()  
        
        if os.path.isdir(results_fp):
            filepath = os.path.join(results_fp, str(start_time))
        else:
            filepath = results_fp

        # [修改] 这里改用 _obs.txt 以匹配你的需求
        files = {
            'inputs': open(f'{filepath}_inputs.txt', 'w'),
            'behaviors': open(f'{filepath}_behaviors.txt', 'w'),
            'cells': open(f'{filepath}_cells.txt', 'w'),
            'logs': open(f'{filepath}_logs.txt', 'w'),
            'final_states': open(f'{filepath}_final_states.txt', 'w'),
            'obs': open(f'{filepath}_obs.txt', 'w') # [新增] 轨迹记录文件
        }

        print(f"Starting Initialization ({init_budget} samples)...")
        inputs, behaviors, acc_rewards, oracles = [], [], [], []
        discovery_times = [] 
        
        # --- 1. Initialization Phase ---
        for _ in tqdm.tqdm(range(init_budget)):
            if time.time() - start_time > time_budget_sec: break
            
            inp = self.generate_random_input()
            # [修改] 接收 traj
            rew, oracle, beh, fs, traj, _ = execute_policy(inp, model, env_seed)
            
            dt = time.time() - start_time
            discovery_times.append(dt)

            inputs.append(inp)
            behaviors.append(beh)
            acc_rewards.append(rew)
            oracles.append(oracle)
            
            # [修改] 使用自定义格式保存轨迹，Generation = 0
            self.save_trajectory_log(files['obs'], 0, inp, oracle, traj)

        behaviors = np.array(behaviors)
        self.xedges, self.yedges = get_edges(env_seed, self.descriptors)
        
        for i in range(len(inputs)):
            cell = compute_cell(behaviors[i], self.xedges, self.yedges).tolist()
            self.update_cell(cell, inputs[i], acc_rewards[i], oracles[i], behaviors[i], 0, discovery_times[i])
            
            np.savetxt(files['inputs'], inputs[i].reshape(1, -1), fmt='%f', delimiter=',')
            np.savetxt(files['behaviors'], behaviors[i].reshape(1, -1), delimiter=',')
            np.savetxt(files['cells'], np.array(cell).reshape(1, -1), fmt='%d', delimiter=',')

        print("Starting Fuzzing Loop...")
        pbar = tqdm.tqdm()
        
        # --- 2. Fuzzing Loop ---
        while time.time() - start_time < time_budget_sec:
            cell_idx = self.select_cell()
            parent_inp, parent_cnt = self.select_input(cell_idx)
            
            mutated_inp = self.mutate(parent_inp)
            # [修改] 接收 traj
            rew, oracle, beh, fs, traj, _ = execute_policy(mutated_inp, model, env_seed)
            
            current_discovery_time = time.time() - start_time
            
            # 记录当前代数
            current_generation = parent_cnt + 1

            cell = compute_cell(beh, self.xedges, self.yedges).tolist()
            self.update_cell(cell, mutated_inp, rew, oracle, beh, current_generation, current_discovery_time)
            
            # [修改] 保存轨迹，Generation = current_generation
            self.save_trajectory_log(files['obs'], current_generation, mutated_inp, oracle, traj)

            np.savetxt(files['inputs'], mutated_inp.reshape(1, -1), fmt='%f', delimiter=',')
            np.savetxt(files['behaviors'], beh.reshape(1, -1), delimiter=',')
            np.savetxt(files['cells'], np.array(cell).reshape(1, -1), fmt='%d', delimiter=',')
            
            pbar.update(1)
            
        pbar.close()
        for f in files.values(): f.close()
        self.save_state(filepath)
        print("Experiment Completed.")