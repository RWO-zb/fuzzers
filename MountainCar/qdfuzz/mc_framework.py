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
        # Each entry: (input, performance, oracle, behavior, mutation_count, discovery_time, seed_id)
        self.cells_data = []

        self.config = {
            'rand_seed': rand_seed,
            'cell_granularity': cell_granularity,
            'name': kwargs.get('name', 'MC-MAP-Elites')
        }

    def save_state(self, filepath: str):
        """Save the MAP-Elites archive to CSV and config to JSON."""
        cell_dfs = []
        base_cols = ['score', 'is_faulty', 'cell_index'] + [f'cell{i}' for i in range(2)] + [f'behavior{i}' for i in range(2)]
        all_cols = base_cols + ['input', 'mutation_count', 'discovery_time', 'seed_id']

        for i, cell_data in enumerate(self.cells_data):
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
        """Select the best-performing input from a cell. Returns (input, mutation_count, seed_id)."""
        scores = [x[1] for x in self.cells_data[index]]
        best_idx = int(np.argmin(scores))
        selected = self.cells_data[index][best_idx]
        return selected[0], selected[4], selected[6]

    def select_cell(self):
        """Randomly select a cell index from the archive."""
        return int(self.rng.integers(0, len(self.cells)))

    def update_cell(self, cell, inp, perf, is_faulty, beh, cnt, discovery_time, seed_id):
        """Insert a new entry into the archive. Creates the cell if it doesn't exist."""
        try:
            idx = self.cells.index(cell)
            self.cells_data[idx].append((inp, perf, is_faulty, beh, cnt, discovery_time, seed_id))
        except ValueError:
            self.cells.append(cell)
            self.cells_data.append([(inp, perf, is_faulty, beh, cnt, discovery_time, seed_id)])
            idx = len(self.cells) - 1
        return idx

    def mutate(self, input_vec):
        """Apply Gaussian noise mutation, clamping position to [-0.6, -0.4] and velocity to 0."""
        noise = self.rng.normal(0, 0.05, size=input_vec.shape)
        mutated = input_vec + noise
        mutated[0] = np.clip(mutated[0], -0.6, -0.4)
        mutated[1] = 0.0 
        return mutated.astype(np.float32)

    def generate_random_input(self):
        """Generate a random initial state with position in [-0.6, -0.4] and zero velocity."""
        pos = self.rng.uniform(-0.6, -0.4)
        return np.array([pos, 0.0], dtype=np.float32)

    def save_trajectory_log(self, file_handle, generation, input_vec, is_faulty, trajectory):
        """Write a single test case header and its observation trajectory to the obs log file."""
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
        Run the MAP-Elites fuzzing pipeline.

        Args:
            model: The trained RL policy to test.
            env_seed: Seed for the Gymnasium environment.
            results_fp: Output file path prefix for all result files.
            init_budget: Number of random samples for the initialization phase.
            time_budget_hours: Wall-clock time limit in hours. None means unlimited.
            max_samples: Maximum total oracle calls (init + fuzz). None means unlimited.
        Note: At least one of time_budget_hours or max_samples should be provided.
        """
        
        # --- 1. Budget setup ---
        start_time = time.time()
        time_limit_sec = (time_budget_hours * 3600) if time_budget_hours is not None else float('inf')
        sample_limit = max_samples if max_samples is not None else float('inf')
        total_executions = 0

        def is_budget_exhausted():
            """Check if either time or sample budget is exceeded."""
            time_used = time.time() - start_time
            if time_used >= time_limit_sec:
                return True, "Time Budget Exceeded"
            if total_executions >= sample_limit:
                return True, "Sample Budget Exceeded"
            return False, ""

        # --- 2. Output file preparation ---
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
        
        # --- 3. Initialization Phase: populate the archive with random inputs ---
        pbar_init = tqdm.tqdm(total=init_budget, desc="Init Phase")
        
        while len(inputs) < init_budget:
            exhausted, reason = is_budget_exhausted()
            if exhausted:
                print(f"\n[Stopping] Initialization stopped: {reason}")
                break
            
            inp = self.generate_random_input()
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

        # Early exit if budget exhausted during initialization
        exhausted, _ = is_budget_exhausted()
        if exhausted and len(inputs) < init_budget:
             print("Budget exhausted during initialization. Skipping fuzzing loop.")
             if len(inputs) > 0:
                behaviors = np.array(behaviors)
                self.xedges, self.yedges = get_edges(env_seed, self.descriptors)
                for i in range(len(inputs)):
                    cell = compute_cell(behaviors[i], self.xedges, self.yedges).tolist()
                    self.update_cell(cell, inputs[i], acc_rewards[i], oracles[i], behaviors[i], 0, discovery_times[i], seed_id=i)
             for f in files.values(): f.close()
             self.save_state(filepath)
             return

        # Build the initial MAP-Elites archive
        behaviors = np.array(behaviors)
        self.xedges, self.yedges = get_edges(env_seed, self.descriptors)
        
        for i in range(len(inputs)):
            cell = compute_cell(behaviors[i], self.xedges, self.yedges).tolist()
            self.update_cell(cell, inputs[i], acc_rewards[i], oracles[i], behaviors[i], 0, discovery_times[i], seed_id=i)
            
            np.savetxt(files['inputs'], inputs[i].reshape(1, -1), fmt='%f', delimiter=',')
            np.savetxt(files['behaviors'], behaviors[i].reshape(1, -1), delimiter=',')
            np.savetxt(files['cells'], np.array(cell).reshape(1, -1), fmt='%d', delimiter=',')

        print("Starting Fuzzing Loop...")
        pbar = tqdm.tqdm(desc="Fuzzing Phase")
        
        # --- 4. Fuzzing Loop: select, mutate, evaluate, and update the archive ---
        while True:
            exhausted, reason = is_budget_exhausted()
            if exhausted:
                print(f"\n[Stopping] Fuzzing stopped: {reason}")
                break

            cell_idx = self.select_cell()
            parent_inp, parent_cnt, parent_seed_id = self.select_input(cell_idx)
            
            mutated_inp = self.mutate(parent_inp)
            rew, oracle, beh, fs, traj, _ = execute_policy(mutated_inp, model, env_seed)
            total_executions += 1
            
            current_discovery_time = time.time() - start_time
            current_generation = parent_cnt + 1

            cell = compute_cell(beh, self.xedges, self.yedges).tolist()
            self.update_cell(cell, mutated_inp, rew, oracle, beh, current_generation, current_discovery_time, seed_id=parent_seed_id)
            
            self.save_trajectory_log(files['obs'], current_generation, mutated_inp, oracle, traj)

            np.savetxt(files['inputs'], mutated_inp.reshape(1, -1), fmt='%f', delimiter=',')
            np.savetxt(files['behaviors'], beh.reshape(1, -1), delimiter=',')
            np.savetxt(files['cells'], np.array(cell).reshape(1, -1), fmt='%d', delimiter=',')
            
            pbar.update(1)
            pbar.set_postfix({'Execs': total_executions, 'Crashes': np.sum([1 for c in self.cells_data for x in c if x[2]])})
            
        pbar.close()
        for f in files.values(): f.close()
        self.save_state(filepath)
        print(f"Experiment Completed. Total Executions: {total_executions}")