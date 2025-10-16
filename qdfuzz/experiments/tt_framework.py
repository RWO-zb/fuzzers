import json
import os
import time
import torch
import tqdm
import numpy as np
import pandas as pd

from typing import List
from common import EXPERIMENT_SEEDS
import tt_common as tt


class Framework:
    def __init__(self, rand_seed: int, **kwargs) -> None:
        self.version = 'random'
        self.rand_seed = rand_seed
        self.rng: np.random.Generator = np.random.default_rng(rand_seed)
        self.creation_time = time.time()

        self.loaded = False
        self.test_budget = None
        self.init_budget = None

        # as indices
        self.last_cell_selected = None
        self.last_cell_updated = None

        # data structure consists of a list of cells (list of integers) and a list of list of test results
        self.cells: list[list[int]] = []
        # the test case results for each cell explored (input, acc. reward, quality, behavior)
        self.cells_data: list[list[tuple[np.ndarray, float, bool, np.ndarray]]] = []

        self.config = {
            'rand_seed': self.rand_seed,
            'use_case': 'Taxi'
        }

        # kwargs (name to include in the experimental configuration etc.)
        self.name = kwargs.get('name')
        if self.name is not None:
            self.config['name'] = self.name
        else:
            self.config['name'] = self.version

        self._env = tt.get_taxi_env()


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
        The resulting data is a .csv file export of a DataFrame and a .npy file of the inputs.
        Both data shares the same order, which is not temporal (logs are though) but results from iterating over the results for each cell.
        '''
        assert len(self.cells) > 0, 'No data to save.'
        cell_size = len(self.cells[0])
        behavior_size = len(self.cells_data[0][0][-1])
        cell_dfs = []
        for i, cell_data in enumerate(self.cells_data):
            # a record consist of a score, the oracle result, the cell index, the cell and behavior point
            cell_dfs.append(
                pd.DataFrame.from_records(
                    data=[[score, is_faulty, i] + self.cells[i] + behavior.tolist() for (_input, score, is_faulty, behavior) in cell_data],
                    columns=['score', 'is_faulty', 'cell_index'] + [f'cell{i}' for i in range(cell_size)] + [f'behavior{i}' for i in range(behavior_size)]
                    )
                )
        pd.concat(cell_dfs, ignore_index=True).to_csv(f'{filepath}_data.csv', index=0)
        # saves the inputs in a .npy file
        np.save(f'{filepath}_inputs.npy', np.concatenate([np.array(list(map(lambda x: x[0], cell_data))) for cell_data in self.cells_data]))
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
        '''Loads a state of an instance to resume testing and returns the number of test cases loaded.'''
        inputs_fp, df_fp = f'{filepath}_inputs.npy', f'{filepath}_data.csv'

        assert os.path.exists(inputs_fp) and os.path.exists(df_fp), 'files are missing.'
        self.cells = []
        self.cells_data = []

        inputs = np.load(inputs_fp)
        df = pd.read_csv(df_fp)
        assert len(inputs) == len(df)

        # removes 1 because of cell_index column
        bs_dim = len([c for c in df.columns.to_list() if c.startswith('cell')]) - 1
        assert bs_dim > 0

        for i, row in df.iterrows():
            row_data = row.tolist()
            cell, input, performance, is_faulty, behavior = row_data[3:3 + bs_dim], inputs[i], row_data[0], row_data[1], row_data[3 + bs_dim:]
            self.update_cell(cell, input, performance, is_faulty, np.array(behavior))

        self.load_random_state(filepath)
        self.load_configuration(filepath)
        self.loaded = True
        return len(df)


    def select_input(self, index: int):
        '''Samples from the indexed cell the next input.'''
        input_index: int = self.rng.integers(0, len(self.cells_data[index]))
        return self.cells_data[index][input_index][0]


    def select_cell(self):
        '''Selects the cell for the next search iteration.'''
        return int(self.rng.integers(0, len(self.cells)))


    def update_cell(self, cell: List[int], input: np.ndarray, performance: float, is_faulty: bool, behavior: np.ndarray):
        '''
        Records the execution result to the corresponding cell.
        It returns the index of the cell updated.
        '''
        index = None
        try:
            index = self.cells.index(cell)
            self.cells_data[index].append((input, performance, is_faulty, behavior))
        except ValueError:
            self.cells.append(cell)
            self.cells_data.append([(input, performance, is_faulty, behavior)])
        finally:
            assert len(self.cells) == len(self.cells_data), 'inconsistent cells and cells_data lists!'
            self.last_cell_updated = index if index is not None else (len(self.cells) - 1)
        return self.last_cell_updated


    def test_policy(self, model: tt.TestAgent,
                    test_budget: int,
                    init_budget: int,
                    results_fp: str,
                    disable_pbar: bool = False):

        assert test_budget > init_budget
        self.test_budget = test_budget
        self.config['test_budget'] = self.test_budget
        self.init_budget = init_budget
        self.config['init_budget'] = self.init_budget

        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp

        behaviors_buffer = open(f'{filepath}_behaviors.txt', 'w', buffering=1)
        final_states_buffer = open(f'{filepath}_final_states.txt', 'w', buffering=1)
        inputs_buffer = open(f'{filepath}_inputs.txt', 'w', buffering=1)
        cells_buffer = open(f'{filepath}_cells.txt', 'w', buffering=1)
        logs_buffer = open(f'{filepath}_logs.txt', 'w', buffering=1)


        time_budget = min(12, test_budget) * 3600
        executions_budget = test_budget - init_budget if test_budget > 12 else 10000
        print(f'Time budget of {(time_budget / 60):.2f} minutes; bound to {executions_budget} executions.')

        bs = tt.BehaviorSpace()

        inputs: List[np.ndarray] = []
        behaviors = []
        final_states: List[int] = []
        acc_rewards: List[float] = []
        oracles: List[bool] = []
        testing_start_time = time.time()
        execution_times = []

        # ensures solutions are always new
        evaluated_solutions = []

        for _ in tqdm.tqdm(range(init_budget), disable=disable_pbar):
            input: np.ndarray = tt.generate_input(self.rng)
            if not (input.tolist() in evaluated_solutions):
                evaluated_solutions.append(input.tolist())
            t0 = time.time()
            episode_reward, oracle, feature, fs, _ = tt.execute_policy(model, input, self._env)
            t1 = time.time()
            execution_times.append(t1 - t0)

            behavior = bs.compute_behavior(feature)

            inputs.append(input)
            behaviors.append(behavior)
            final_states.append(fs)
            acc_rewards.append(episode_reward)
            oracles.append(oracle)
        behaviors = np.array(behaviors)

        print(f'{len(evaluated_solutions)} distinct inputs evaluated after initialization.')

        for i in range(init_budget):
            behavior = behaviors[i]
            cell = bs.compute_cell(behavior)
            mutated_input_index = self.update_cell(cell, inputs[i], acc_rewards[i], oracles[i], behavior)
            print(f'episode_reward: {acc_rewards[i]}, oracle: {float(oracles[i])}, cell_selected_index: -1, cell_updated_index: {mutated_input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, inputs[i].reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
            final_states_buffer.write(f'{final_states[i]}\n')

        start_time = time.time()
        current_time = time.time()
        nb_executions = 0
        pbar = tqdm.tqdm(total=executions_budget, disable=disable_pbar)

        while (current_time - start_time < time_budget) and (nb_executions < executions_budget):
            cell_index = self.select_cell()
            self.last_cell_selected = cell_index
            input = self.select_input(cell_index)

            # +/- fixes mutation operator (but mutants still might be already evaluated...)
            attempts = 1
            while attempts < 40:
                mutated_input = tt.mutate(input, self.rng)
                tmp = mutated_input.tolist()
                if not (tmp in evaluated_solutions):
                    evaluated_solutions.append(tmp)
                    break
                else:
                    attempts += 1

            t0 = time.time()
            episode_reward, oracle, feature, fs, _ = tt.execute_policy(model, mutated_input, self._env)
            t1 = time.time()
            execution_times.append(t1 - t0)
            behavior, cell = bs.describe(feature)

            mutated_input_index = self.update_cell(cell, mutated_input, episode_reward, oracle, behavior)
            print(f'episode_reward: {episode_reward}, oracle: {float(oracle)}, cell_selected_index: {cell_index}, cell_updated_index: {mutated_input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, mutated_input.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
            final_states_buffer.write(f'{fs}\n')
            current_time = time.time()
            nb_executions += 1
            pbar.update(1)

        testing_end_time = time.time()
        self.config['testing_start_time'] = testing_start_time
        self.config['testing_end_time'] = testing_end_time
        self.config['testing_time'] = testing_end_time - testing_start_time
        self.config['total_execution_time'] = sum(execution_times)

        pbar.close()
        behaviors_buffer.close()
        inputs_buffer.close()
        cells_buffer.close()
        logs_buffer.close()
        final_states_buffer.close()
        self.save_state(filepath)


    def random_testing(self, model: tt.TestAgent,
                    test_budget: int,
                    results_fp: str,
                    disable_pbar: bool = False):
        '''Random testing loop baseline.'''

        self.test_budget = test_budget
        self.config['test_budget'] = self.test_budget

        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp

        behaviors_buffer = open(f'{filepath}_behaviors.txt', 'w', buffering=1)
        final_states_buffer = open(f'{filepath}_final_states.txt', 'w', buffering=1)
        inputs_buffer = open(f'{filepath}_inputs.txt', 'w', buffering=1)
        cells_buffer = open(f'{filepath}_cells.txt', 'w', buffering=1)
        logs_buffer = open(f'{filepath}_logs.txt', 'w', buffering=1)


        time_budget = min(12, test_budget) * 3600
        executions_budget = test_budget if test_budget > 12 else 10000
        print(f'Time budget of {(time_budget / 60):.2f} minutes; bound to {executions_budget} executions.')

        bs = tt.BehaviorSpace()

        execution_times = []

        start_time = time.time()
        current_time = time.time()
        nb_executions = 0
        pbar = tqdm.tqdm(total=executions_budget, disable=disable_pbar)

        while (current_time - start_time < time_budget) and (nb_executions < executions_budget):
            input: np.ndarray = tt.generate_input(self.rng)
            t0 = time.time()
            episode_reward, oracle, feature, fs, _ = tt.execute_policy(model, input, self._env)
            t1 = time.time()
            execution_times.append(t1 - t0)
            behavior, cell = bs.describe(feature)

            input_index = self.update_cell(cell, input, episode_reward, oracle, behavior)
            print(f'episode_reward: {episode_reward}, oracle: {float(oracle)}, cell_selected_index: -1, cell_updated_index: {input_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, input.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
            final_states_buffer.write(f'{fs}\n')
            current_time = time.time()
            nb_executions += 1
            pbar.update(1)

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
        self.save_state(filepath)


    def novelty_search(self, model: tt.TestAgent,
                    pop_size: int,
                    nb_iterations: int,
                    k: int,
                    nov_threshold: float,
                    results_fp: str,
                    disable_pbar: bool = False):

        self.config['pop_size'] = pop_size
        self.config['nb_iterations'] = nb_iterations
        self.config['test_budget'] = pop_size * nb_iterations
        self.config['nov_threshold'] = nov_threshold
        self.config['k'] = k

        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp

        # to collect the data during the search, i.e., every model execution
        behaviors_buffer = open(f'{filepath}_behaviors.txt', 'w', buffering=1)
        final_states_buffer = open(f'{filepath}_final_states.txt', 'w', buffering=1)
        inputs_buffer = open(f'{filepath}_inputs.txt', 'w', buffering=1)
        cells_buffer = open(f'{filepath}_cells.txt', 'w', buffering=1)
        logs_buffer = open(f'{filepath}_logs.txt', 'w', buffering=1)

        bs = tt.BehaviorSpace()

        # ensures solutions are always new
        evaluated_solutions = []

        # helpers 1: recording the executions during each iteration
        def record(input: np.ndarray, reward: float, oracle: bool, feature: np.ndarray, final_state: int) -> None:
            behavior, cell = bs.describe(feature)
            updated_cell_index = self.update_cell(cell, input, reward, oracle, behavior)
            # parent's cell is not logged
            print(f'episode_reward: {reward}, oracle: {float(oracle)}, cell_updated_index: {updated_cell_index}, nb_cells: {len(self.cells)}', file=logs_buffer)
            np.savetxt(inputs_buffer, input.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
            final_states_buffer.write(f'{final_state}\n')
            return behavior
        # helpers 2: evaluates a batch of individuals
        def evaluate(individuals: np.ndarray) -> np.ndarray:
            behaviors = []
            for ind in individuals:
                r, o, b, fs, _ = tt.execute_policy(model, ind, self._env)
                behavior = record(ind, r, o, b, fs)
                behaviors.append(behavior)
            return np.array(behaviors)
        # helper 3: mutates a batch of individuals
        def mutate(inputs: np.ndarray):
            mutants = []
            for input in inputs:
                attempts = 1
                while attempts < 40:
                    mutated_input = tt.mutate(input, self.rng)
                    tmp = mutated_input.tolist()
                    if not (tmp in evaluated_solutions):
                        evaluated_solutions.append(tmp)
                        break
                    else:
                        attempts += 1
                mutants.append(mutated_input)
            return np.array(mutants)

        # ns logs
        ns_logs_buffer = open(f'{filepath}_ns_logs.txt', 'w', buffering=1)
        nov_scores_buffer = open(f'{filepath}_nov_scores.txt', 'w', buffering=1)
        # initial population and novelty archive
        from novelty_search import NoveltyArchive
        pop = []
        for _ in range(pop_size):
            input: np.ndarray = tt.generate_input(self.rng)
            if not (input.tolist() in evaluated_solutions):
                evaluated_solutions.append(input.tolist())
            pop.append(input)
        # print(f'{len(evaluated_solutions)} distinct inputs in the initial population ({(100*len(evaluated_solutions)/pop_size):0.2f}%)')
        pop_behaviors = evaluate(pop)
        nov_archive = NoveltyArchive(pop_behaviors, k, nov_threshold)
        pop_nov_scores = nov_archive.score(pop_behaviors)
        [np.savetxt(nov_scores_buffer, s.reshape(1, -1), delimiter=',') for s in pop_nov_scores]
        # novelty search looop
        print(f'iteration: 0, archive_size: {nov_archive.size()}, archive_sparseness: {nov_archive.archive_sparseness():0.5f}, evaluated_solutions: {len(evaluated_solutions)}', file=ns_logs_buffer)
        for i in tqdm.tqdm(range(1, nb_iterations), disable=disable_pbar):
            # 1. generates offspring
            offspring = mutate(pop)
            # 1. evaluates the offspring
            offspring_behaviors = evaluate(offspring)
            # 1. novelty scores of the offspring w.r.t the archive and the population
            offspring_nov_scores = nov_archive.score(offspring_behaviors, pop_behaviors)

            # 2. selects the most novel individuals to form the new population
            joined_pop = np.vstack([pop, offspring])
            joined_scores = np.hstack([pop_nov_scores, offspring_nov_scores])
            median_score = np.median(joined_scores)

            # 3. updates the archive
            _updated, _offspring_indices = nov_archive.update3(offspring_behaviors)

             # 4. updates the population and their data
            mask = (joined_scores >= median_score)

            pop = joined_pop[mask].copy()
            pop_behaviors = np.vstack([pop_behaviors, offspring_behaviors])[mask]
            pop_nov_scores = nov_archive.score(pop_behaviors)
            if len(pop) > pop_size:
                pop = pop[:pop_size]
                pop_behaviors = pop_behaviors[:pop_size]
                pop_nov_scores = pop_nov_scores[:pop_size]


            # assert len(pop) == pop_size, (len(pop), pop.shape)
            # assert len(pop_behaviors) == pop_size, (len(pop), pop.shape)
            # assert len(pop_nov_scores) == pop_size, (len(pop), pop.shape)
            [np.savetxt(nov_scores_buffer, s.reshape(1, -1), delimiter=',') for s in pop_nov_scores]
            print(f'iteration: {i}, archive_size: {nov_archive.size()}, archive_sparseness: {nov_archive.archive_sparseness():0.5f}, evaluated_solutions: {len(evaluated_solutions)}', file=ns_logs_buffer)

        behaviors_buffer.close()
        inputs_buffer.close()
        cells_buffer.close()
        logs_buffer.close()
        final_states_buffer.close()
        self.save_state(filepath)


class MAPElitesFramework(Framework):
    def __init__(self, rand_seed: int, **kwargs) -> None:
        if kwargs.get('name') is None:
            kwargs['name'] = 'MAP-Elites'
        super().__init__(rand_seed, **kwargs)


    def select_input(self, index: int):
        scores = list(map(lambda x: x[1], self.cells_data[index]))
        # the best performing input is one whose score is the minimum, since it corresponds to the accumulated reward.
        best_performer_index = int(np.argmin(scores))
        return self.cells_data[index][best_performer_index][0]


if __name__ == '__main__':
    torch.set_num_threads(1)
    main_seed = 2021
    env_seed = 0
    model = tt.load_taxi_model()

    # experimental parameters
    test_budget = 5000
    init_budget = 1000
    cell_granularity = 50

    population_size, nb_iterations = 100, 50
    k = 3
    novelty_threshold = 0.9

    results_fp = 'results/tt'
    if not os.path.isdir(results_fp):
        os.mkdir(results_fp)

    for seed in EXPERIMENT_SEEDS:
        f = Framework(seed, name='Random Testing')
        f.random_testing(model, test_budget, results_fp)
        f = MAPElitesFramework(seed, name='MAP-Elites')
        f.test_policy(model, test_budget, init_budget, results_fp)

    results_fp = 'results/ttns'
    if not os.path.isdir(results_fp):
        os.mkdir(results_fp)

    # Novelty Search
    for seed in EXPERIMENT_SEEDS:
        f = Framework(seed, name='Novelty Search')
        f.novelty_search(
            model,
            population_size,
            nb_iterations,
            k,
            novelty_threshold,
            results_fp
        )