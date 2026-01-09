import json
import os
import time
import tqdm
import numpy as np
import pandas as pd
import carla
from typing import List

from common import compute_cell, get_edges
from carla_common import load_model, execute_policy, generate_random_individual

class Framework():
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(rand_seed)
        self.granularity = cell_granularity
        self.descriptors = np.array(descriptors)
        
        self.cells = [] 
        # (input_obj, score, is_faulty, behavior, mutation_count, time, post_str)
        self.cells_data = [] 
        
        self.config = {
            'rand_seed': self.rand_seed,
            'cell_granularity': self.granularity,
            'use_case': 'CARLA_QD_GAUSSIAN_PHYSICAL'
        }
        self.name = kwargs.get('name', 'QD-CURE-Gaussian')

    def save_state(self, filepath: str):
        data_rows = []
        for i, cell_list in enumerate(self.cells_data):
            for item in cell_list:
                data_rows.append({
                    'score': item[1],
                    'is_faulty': item[2],
                    'behavior_0': item[3][0],
                    'behavior_1': item[3][1],
                    'mutation_count': item[4],
                    'elapsed_time': item[5],
                    'state_str': item[6] 
                })
        
        if data_rows:
            pd.DataFrame(data_rows).to_csv(f'{filepath}_qd_data.csv', index=False)

    def select_input(self, index: int):
        input_index = self.rng.integers(0, len(self.cells_data[index]))
        selected_data = self.cells_data[index][input_index]
        return selected_data[0], selected_data[4], selected_data[6]

    def select_cell(self):
        return int(self.rng.integers(0, len(self.cells)))

    def update_cell(self, cell, input_obj, score, faulty, behavior, mut_cnt, time, post_str):
        try:
            index = self.cells.index(cell)
            self.cells_data[index].append((input_obj, score, faulty, behavior, mut_cnt, time, post_str))
        except ValueError:
            self.cells.append(cell)
            self.cells_data.append([(input_obj, score, faulty, behavior, mut_cnt, time, post_str)])

    # ================= 核心修改：高斯分布物理变异 =================
    def mutate(self, individual):
        """
        Input: individual tuple (ego_trans, npc_info_list, weather, start_id, target_id)
        Logic: Apply Gaussian noise to x, y, yaw of Ego and x, y of NPCs.
        """
        ego_transform, npc_info, weather, start, target = individual
        
        # 1. 主车变异 (Ego Mutation)
        # 保持 QD 的高斯分布特性 (np.random.normal)
        # 对齐 CURE 的变异对象 (x, y, yaw)
        new_ego = carla.Transform(
            carla.Location(ego_transform.location.x, ego_transform.location.y, ego_transform.location.z),
            carla.Rotation(ego_transform.rotation.pitch, ego_transform.rotation.yaw, ego_transform.rotation.roll)
        )
        
        # 使用高斯噪声，Sigma 设为 0.15 和 5.0 以匹配 CURE 的幅度量级
        new_ego.location.x += self.rng.normal(0, 0.15)
        new_ego.location.y += self.rng.normal(0, 0.15)
        new_ego.rotation.yaw += self.rng.normal(0, 5.0)
        
        # 2. NPC 变异 (NPC Mutation)
        # 对齐 CURE 的变异对象 (x, y)
        new_npcs = []
        for item in npc_info:
            bp_id, t, color, driver_id = item
            new_t = carla.Transform(
                carla.Location(t.location.x, t.location.y, t.location.z),
                carla.Rotation(t.rotation.pitch, t.rotation.yaw, t.rotation.roll)
            )
            
            # 使用高斯噪声，Sigma 设为 0.1
            new_t.location.x += self.rng.normal(0, 0.1)
            new_t.location.y += self.rng.normal(0, 0.1)
            
            new_npcs.append((bp_id, new_t, color, driver_id))
            
        return (new_ego, new_npcs, weather, start, target)
    # ==========================================================

    def test_policy(self, model, env_seed: int, time_budget_hours: int, init_budget: int, results_fp: str):
        filepath = str(model.result_dir / self.name)
        time_budget_seconds = time_budget_hours * 3600
        
        print("Starting Initialization (Phase 1: Valid Seed Collection)...")
        start_time = time.time()
        
        xedges, yedges = get_edges(env_seed, self.descriptors)

        # Phase 1: 初始化 Archive (Modified for filtering)
        # 循环直到找到 init_budget 个合法的种子 (Success且无碰撞)
        valid_seeds = 0
        attempt = 0
        pbar = tqdm.tqdm(total=init_budget, desc="Finding Seeds")
        
        while valid_seeds < init_budget:
            attempt += 1
            # 使用包含尝试次数的种子，确保每次生成都是随机且不同的
            current_seed = env_seed + attempt * 1000
            
            run_name = f"seed_{attempt:04d}" # 日志记录每次尝试
            
            # 生成随机合法的物理场景
            individual = generate_random_individual(model, seed=current_seed)
            
            # 执行策略，接收额外的 stop_reason
            score, faulty, behavior, _, _, post_str, stop_reason = execute_policy(
                individual, model, env_seed, 
                mutation_generation=0, run_name=run_name, phase="Phase1", input_pre="None"
            )
            
            # 过滤逻辑：只有成功的任务 (到达终点) 且无碰撞才作为初始种子
            # CURE 逻辑参考: if res['success'] and not res['collision']
            if stop_reason == "Success" and not faulty:
                cell = compute_cell(behavior, xedges, yedges).tolist()
                self.update_cell(cell, individual, score, faulty, behavior, 0, time.time()-start_time, post_str)
                valid_seeds += 1
                pbar.update(1)
                # print(f"Found valid seed {valid_seeds}/{init_budget}: {run_name}")
            else:
                # 失败的尝试会被记录在CSV中(由execute_policy负责)，但不加入MAP-Elites Archive
                pass

        pbar.close()

        print(f"Starting Fuzzing (Phase 2)... Initialized with {valid_seeds} valid seeds.")
        fuzz_count = 0
        fuzz_start_time = time.time()
        
        pbar = tqdm.tqdm()
        while (time.time() - fuzz_start_time < time_budget_seconds):
            fuzz_count += 1
            run_name = f"fuzz_{fuzz_count:04d}"
            
            if len(self.cells) == 0: continue
            
            # Phase 2: 选择 + 变异
            cell_idx = self.select_cell()
            parent_ind, parent_gen, parent_post_str = self.select_input(cell_idx)
            
            # 调用高斯物理变异
            child_ind = self.mutate(parent_ind)
            new_gen = parent_gen + 1
            
            # 注意：此处也需要接收 7 个返回值
            score, faulty, behavior, _, _, child_post_str, stop_reason = execute_policy(
                child_ind, model, env_seed,
                mutation_generation=new_gen, run_name=run_name, phase="Phase2", input_pre=parent_post_str
            )
            
            cell = compute_cell(behavior, xedges, yedges).tolist()
            self.update_cell(cell, child_ind, score, faulty, behavior, new_gen, time.time()-start_time, child_post_str)
            pbar.update(1)
            
        self.save_state(filepath)

class MAPElitesFramework(Framework):
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        kwargs['name'] = 'MAP-Elites-Physical-Gaussian'
        super().__init__(rand_seed, cell_granularity, descriptors, **kwargs)

    def select_input(self, index: int):
        scores = list(map(lambda x: x[1], self.cells_data[index]))
        best_idx = int(np.argmax(scores))
        selected = self.cells_data[index][best_idx]
        return selected[0], selected[4], selected[6]