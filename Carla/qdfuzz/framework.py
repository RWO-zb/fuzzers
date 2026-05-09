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
    """
    Base framework for Quality Diversity (QD) fuzzing in CARLA.
    """
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(rand_seed)
        self.granularity = cell_granularity
        self.descriptors = np.array(descriptors)
        
        self.cells = [] 
        # Stores: (input_obj, score, is_faulty, behavior, mutation_count, time, post_str)
        self.cells_data = [] 
        
        self.config = {
            'rand_seed': self.rand_seed,
            'cell_granularity': self.granularity,
            'use_case': 'CARLA_QD_GAUSSIAN_PHYSICAL'
        }
        self.name = kwargs.get('name', 'QD-CURE-Gaussian')

    def save_state(self, filepath: str):
        """
        Saves the current state of the QD archive to a CSV file.
        """
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
        """
        Randomly selects an individual from a specific cell.
        """
        input_index = self.rng.integers(0, len(self.cells_data[index]))
        selected_data = self.cells_data[index][input_index]
        return selected_data[0], selected_data[4], selected_data[6]

    def select_cell(self):
        """
        Randomly selects a non-empty cell index from the archive.
        """
        return int(self.rng.integers(0, len(self.cells)))

    def update_cell(self, cell, input_obj, score, faulty, behavior, mut_cnt, time, post_str):
        """
        Updates the archive with a new individual. Creates a new cell if it doesn't exist.
        """
        try:
            index = self.cells.index(cell)
            self.cells_data[index].append((input_obj, score, faulty, behavior, mut_cnt, time, post_str))
        except ValueError:
            self.cells.append(cell)
            self.cells_data.append([(input_obj, score, faulty, behavior, mut_cnt, time, post_str)])

    def mutate(self, individual):
        """
        Applies Gaussian physical mutation to the individual (Ego and NPC transforms).
        
        Input: individual tuple (ego_trans, npc_info_list, weather, start_id, target_id)
        Logic: Apply Gaussian noise to x, y, yaw of Ego and x, y of NPCs.
        """
        ego_transform, npc_info, weather, start, target = individual
        
        # 1. Ego Mutation
        # Maintain QD's Gaussian distribution characteristics (np.random.normal)
        # Align with CURE's mutation targets (x, y, yaw)
        new_ego = carla.Transform(
            carla.Location(ego_transform.location.x, ego_transform.location.y, ego_transform.location.z),
            carla.Rotation(ego_transform.rotation.pitch, ego_transform.rotation.yaw, ego_transform.rotation.roll)
        )
        
        # Use Gaussian noise with Sigma set to 0.15 and 5.0 to match CURE's magnitude
        new_ego.location.x += self.rng.normal(0, 0.15)
        new_ego.location.y += self.rng.normal(0, 0.15)
        new_ego.rotation.yaw += self.rng.normal(0, 5.0)
        
        # 2. NPC Mutation
        # Align with CURE's mutation targets (x, y)
        new_npcs = []
        for item in npc_info:
            bp_id, t, color, driver_id = item
            new_t = carla.Transform(
                carla.Location(t.location.x, t.location.y, t.location.z),
                carla.Rotation(t.rotation.pitch, t.rotation.yaw, t.rotation.roll)
            )
            
            # Use Gaussian noise with Sigma set to 0.1
            new_t.location.x += self.rng.normal(0, 0.1)
            new_t.location.y += self.rng.normal(0, 0.1)
            
            new_npcs.append((bp_id, new_t, color, driver_id))
            
        return (new_ego, new_npcs, weather, start, target)

    def test_policy(self, model, env_seed: int, time_budget_hours: int, init_budget: int, results_fp: str):
        """
        Main execution loop for testing policy using QD-fuzzing.
        """
        filepath = str(model.result_dir / self.name)
        time_budget_seconds = time_budget_hours * 3600
        
        print("Starting Initialization (Phase 1: Valid Seed Collection)...")
        start_time = time.time()
        
        xedges, yedges = get_edges(env_seed, self.descriptors)

        # Pre-generate all unique task and weather combinations
        all_combinations = []
        for t_idx in range(len(model.tasks)):
            for w_id in model.weathers:
                all_combinations.append((t_idx, w_id))
        
        # Deterministically shuffle the combination order using the initialized RNG
        self.rng.shuffle(all_combinations)
        
        combo_iterator = iter(all_combinations)
        total_combos = len(all_combinations)
        
        print(f"Generated {total_combos} unique Task/Weather combinations for initialization.")

        # Phase 1: Archive Initialization
        # Loop until init_budget valid seeds (Success and no collision) are found
        valid_seeds = 0
        attempt = 0
        pbar = tqdm.tqdm(total=init_budget, desc="Finding Seeds")
        
        while valid_seeds < init_budget:
            # Attempt to get the next unique combination
            try:
                t_idx, w_id = next(combo_iterator)
            except StopIteration:
                # Stop initialization early if all unique combinations are exhausted
                print(f"\n[Warning] Exhausted all {total_combos} unique combinations. Stopping initialization early.")
                break

            attempt += 1
            # Use seed with attempt count to ensure diverse NPC distributions
            current_seed = env_seed + attempt * 1000
            
            run_name = f"seed_{attempt:04d}" # Log each attempt
            
            # Generate random valid physical scenes using assigned t_idx and w_id
            individual = generate_random_individual(model, seed=current_seed, task_idx=t_idx, weather_id=w_id)
            
            # Execute policy simulation
            score, faulty, behavior, _, _, post_str, stop_reason = execute_policy(
                individual, model, env_seed, 
                mutation_generation=0, run_name=run_name, phase="Phase1", input_pre="None"
            )
            
            # Filter logic: only successful tasks are added to the archive
            if stop_reason == "Success" and not faulty:
                cell = compute_cell(behavior, xedges, yedges).tolist()
                self.update_cell(cell, individual, score, faulty, behavior, 0, time.time()-start_time, post_str)
                valid_seeds += 1
                pbar.update(1)
            else:
                # Failed attempts are logged but not added to the archive
                pass

        pbar.close()

        print(f"Starting Fuzzing (Phase 2)... Initialized with {valid_seeds} valid seeds.")
        fuzz_count = 0
        fuzz_start_time = time.time()
        
        pbar = tqdm.tqdm()
        while (time.time() - fuzz_start_time < time_budget_seconds):
            fuzz_count += 1
            run_name = f"fuzz_{fuzz_count:04d}"
            
            if len(self.cells) == 0: 
                print("[Error] No valid seeds in archive. Stopping.")
                break
            
            # Phase 2: Selection + Mutation
            cell_idx = self.select_cell()
            parent_ind, parent_gen, parent_post_str = self.select_input(cell_idx)
            
            # Perform Gaussian physical mutation
            child_ind = self.mutate(parent_ind)
            new_gen = parent_gen + 1
            
            score, faulty, behavior, _, _, child_post_str, stop_reason = execute_policy(
                child_ind, model, env_seed,
                mutation_generation=new_gen, run_name=run_name, phase="Phase2", input_pre=parent_post_str
            )
            
            cell = compute_cell(behavior, xedges, yedges).tolist()
            self.update_cell(cell, child_ind, score, faulty, behavior, new_gen, time.time()-start_time, child_post_str)
            pbar.update(1)
            
        self.save_state(filepath)

class MAPElitesFramework(Framework):
    """
    MAP-Elites implementation that selects the best-performing individual in a cell.
    """
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        kwargs['name'] = 'MAP-Elites-Physical-Gaussian'
        super().__init__(rand_seed, cell_granularity, descriptors, **kwargs)

    def select_input(self, index: int):
        """
        Selects the individual with the highest score in the specified cell.
        """
        scores = list(map(lambda x: x[1], self.cells_data[index]))
        best_idx = int(np.argmax(scores))
        selected = self.cells_data[index][best_idx]
        return selected[0], selected[4], selected[6]