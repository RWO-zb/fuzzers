import json
import os
import time
import tqdm
import numpy as np
import pandas as pd
from typing import List

# Change Imports to local
from common import compute_cell, EXPERIMENT_SEEDS, get_edges
from carla_common import load_model, execute_policy

class Framework():
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        self.version = 'random'
        self.rand_seed = rand_seed
        self.rng: np.random.Generator = np.random.default_rng(rand_seed)
        self.creation_time = time.time()
        self.loaded = False
        self.init_budget = None
        self.granularity = cell_granularity
        self.descriptors: np.ndarray = np.array(descriptors)
        
        self.cells: list[list[int]] = []
        # (input, performance, oracle result, behavior, mutation_count, elapsed_time)
        self.cells_data: list[list[tuple[np.ndarray, float, bool, np.ndarray, int, float]]] = []

        self.config = {
            'rand_seed': self.rand_seed,
            'cell_granularity': self.granularity,
            'descriptors': self.descriptors.tolist(),
            'use_case': 'CARLA'
        }
        
        self.name = kwargs.get('name', self.version)
        self.config['name'] = self.name

    def save_state(self, filepath: str):
        cell_dfs = []
        base_columns = ['score', 'is_faulty', 'cell_index'] + [f'cell{i}' for i in range(2)] + [f'behavior{i}' for i in range(2)]
        all_columns = base_columns + ['input', 'mutation_count', 'elapsed_time']

        for i, cell_data in enumerate(self.cells_data):
            cell_dfs.append(
                pd.DataFrame.from_records(
                    data=[[score, is_faulty, i] + self.cells[i] + behavior.tolist() + [json.dumps(_input.tolist())] + [mutation_count] + [elapsed_time]
                          for (_input, score, is_faulty, behavior, mutation_count, elapsed_time) in cell_data],
                    columns=all_columns
                )
            )
        
        if cell_dfs:
            pd.concat(cell_dfs, ignore_index=True).to_csv(f'{filepath}_data.csv', index=0)
        
        f = open(f'{filepath}_config.json', 'w')
        f.write(json.dumps(self.config))
        f.close()

    def select_input(self, index: int):
        input_index: int = self.rng.integers(0, len(self.cells_data[index]))
        selected_data = self.cells_data[index][input_index]
        return selected_data[0], selected_data[4]

    def select_cell(self):
        return int(self.rng.integers(0, len(self.cells)))

    def update_cell(self, cell: List[int], input: np.ndarray, performance: float, is_faulty: bool, behavior: np.ndarray, mutation_count: int, elapsed_time: float):
        index = None
        try:
            index = self.cells.index(cell)
            self.cells_data[index].append((input, performance, is_faulty, behavior, mutation_count, elapsed_time))
        except ValueError:
            self.cells.append(cell)
            self.cells_data.append([(input, performance, is_faulty, behavior, mutation_count, elapsed_time)])
        return index

    def mutate(self, input: np.ndarray) -> np.ndarray:
        # [MODIFIED for CARLA]
        # Input is a continuous vector [0, 1] of size 10
        # We apply Gaussian noise and clip
        mutation_mag = 0.1
        mutation = self.rng.normal(0, mutation_mag, size=input.shape)
        mutated_input = np.clip(input + mutation, 0.0, 1.0)
        return mutated_input

    def test_policy(self, model, env_seed: int, time_budget_hours: int, init_budget: int, results_fp: str):
        self.config['time_budget_hours'] = time_budget_hours
        self.init_budget = init_budget
        
        if os.path.isdir(results_fp):
            filepath = f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp

        time_budget_seconds = time_budget_hours * 3600
        print(f'Time budget of {time_budget_hours} hours.')

        inputs: List[np.ndarray] = []
        behaviors = []
        execution_times = []
        
        testing_start_time = time.time()
        
        # 1. Initialization Phase
        print("Starting initialization phase...")
        for _ in tqdm.tqdm(range(init_budget)):
            if (time.time() - testing_start_time > time_budget_seconds): break
            
            # Generate random input [0, 1] size 10
            input_vec = self.rng.random(size=10)
            
            t0 = time.time()
            episode_reward, oracle, behavior, _, _ = execute_policy(input_vec, model, env_seed, self.descriptors)
            t1 = time.time()
            
            execution_times.append(t1 - t0)
            inputs.append(input_vec)
            behaviors.append(behavior)
            
            # Initial update
            xedges, yedges = get_edges(env_seed, self.descriptors)
            cell = compute_cell(behavior, xedges, yedges).tolist()
            self.update_cell(cell, input_vec, episode_reward, oracle, behavior, 0, t1 - testing_start_time)

        # 2. Fuzzing Phase
        print("Starting fuzzing loop...")
        pbar = tqdm.tqdm()
        
        while (time.time() - testing_start_time < time_budget_seconds):
            cell_index = self.select_cell()
            input_vec, parent_mutation_count = self.select_input(cell_index)
            
            mutated_input = self.mutate(input_vec)
            
            t0 = time.time()
            episode_reward, oracle, behavior, _, _ = execute_policy(mutated_input, model, env_seed, self.descriptors)
            t1 = time.time()
            
            cell = compute_cell(behavior, xedges, yedges).tolist()
            self.update_cell(cell, mutated_input, episode_reward, oracle, behavior, parent_mutation_count + 1, t1 - testing_start_time)
            pbar.update(1)

        pbar.close()
        self.save_state(filepath)
        print("Done.")

class MAPElitesFramework(Framework):
    def __init__(self, rand_seed: int, cell_granularity: int, descriptors: List[int], **kwargs) -> None:
        kwargs['name'] = 'MAP-Elites'
        super().__init__(rand_seed, cell_granularity, descriptors, **kwargs)

    def select_input(self, index: int):
        # Select best performer in cell (Max reward)
        scores = list(map(lambda x: x[1], self.cells_data[index]))
        best_performer_index = int(np.argmax(scores)) # Changed to argmax for Reward
        selected_data = self.cells_data[index][best_performer_index]
        return selected_data[0], selected_data[4]