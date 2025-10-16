import json
import copy
import tqdm
import torch
import os
import time
import numpy as np
import pandas as pd

from typing import List, Dict, Tuple
from stable_baselines3.common.base_class import BaseAlgorithm
from scipy.stats import multivariate_normal

from common import compute_cell, EXPERIMENT_SEEDS
from bw_common import load_model, execute_policy_trajectory, execute_policy, EXPERT_INDICES
from result_analysis import isin


class GridCollection():
    '''
    Collection of 2d behaviors for the Bipedal Walker use-case.
    It is a quick way to run MDPFuzz once while having the 4 collections of results.
    '''

    def __init__(self, filepath: str, descriptors: List[int], edges: np.ndarray, initial_config: Dict = {}) -> None:

        self.fp = filepath

        self.descriptors: np.ndarray = np.array(descriptors)
        # 2d behavior space
        assert len(self.descriptors) == 2
        assert all(self.descriptors < 12) and all(self.descriptors >= 0)

        self.last_cell_updated = None

        # data structure consists of a list of cells (list of integers) and a list of list of test results
        self.cells: list[list[int]] = []
        # the test case results for each cell explored (input, acc. reward, quality, behavior)
        self.cells_data: list[list[tuple[np.ndarray, float, bool, np.ndarray]]] = []

        self.config = initial_config

        # updates the configuration dictionary
        try:
            index = EXPERT_INDICES.index(self.descriptors.tolist())
            self.config['use_case'] = f'Bipedal Walker {index}'
        except ValueError:
            pass

        # edges for the grid
        self.edges = edges[self.descriptors]
        self.xedges: np.ndarray = self.edges[0]
        self.yedges: np.ndarray = self.edges[1]
        self.config['xedges'] = list(self.xedges)
        self.config['yedges'] = list(self.xedges)

        # buffers to record results such as logs etc.
        self.behaviors_buffer = open(f'{self.fp}_behaviors.txt', 'w', buffering=1)
        self.final_states_buffer = open(f'{self.fp}_final_states.txt', 'w', buffering=1)
        self.inputs_buffer = open(f'{self.fp}_inputs.txt', 'w', buffering=1)
        self.cells_buffer = open(f'{self.fp}_cells.txt', 'w', buffering=1)
        self.logs_buffer = open(f'{self.fp}_logs.txt', 'w', buffering=1)


    def save_configuration(self):
        '''Saves the collection metadata.'''
        f = open(f'{self.fp}_config.json', 'w')
        f.write(json.dumps(self.config))
        f.close()


    def save_state(self):
        '''Exports the collection as a .csv file of a DataFrame, numpy array of all solutions and eventually saves metadata of the configuration.'''
        # creates DataFrames and exports them in a single .csv file
        cell_dfs = []
        for i, cell_data in enumerate(self.cells_data):
            cell_dfs.append(
                pd.DataFrame.from_records(
                    data=[[score, is_faulty, i] + self.cells[i] + behavior.tolist() for (_input, score, is_faulty, behavior) in cell_data],
                    columns=['score', 'is_faulty', 'cell_index'] + [f'cell{i}' for i in range(2)] + [f'behavior{i}' for i in range(2)]
                    )
                )
        pd.concat(cell_dfs, ignore_index=True).to_csv(f'{self.fp}_data.csv', index=0)
        # saves the inputs
        np.save(f'{self.fp}_inputs.npy', np.concatenate([np.array(list(map(lambda x: x[0], cell_data))) for cell_data in self.cells_data]))
        # saves the configuration
        self.save_configuration()
        # closes all the buffers
        self.inputs_buffer.close()
        self.behaviors_buffer.close()
        self.cells_buffer.close()
        self.logs_buffer.close()
        self.final_states_buffer.close()


    def update_cell(self, input: np.ndarray, performance: float, is_faulty: bool, feature: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        behavior: np.ndarray = feature[self.descriptors]
        cell: List[int] = compute_cell(behavior, self.xedges, self.yedges).tolist()

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
            return behavior, cell


    def get_cell(self, input: np.ndarray):
        '''Finds the input's data in the collection and return its cell index.'''
        for i in range(len(self.cells_data)):
            inputs = np.array(list(map(lambda x: x[0], self.cells_data[i])))
            if isin(inputs, input).any():
                return i
        # should never happen
        raise ValueError(f'Input has not been found: {input}')


    def record_init_execution(self, input: np.ndarray, performance: float, is_faulty: bool, feature: np.ndarray, final_state: np.ndarray, exec_time: float):
        # updates the collection structure
        behavior, cell = self.update_cell(input, performance, is_faulty, feature)
        # saves numpy arrays
        np.savetxt(self.inputs_buffer, input.reshape(1, -1), fmt='%1.0f', delimiter=',')
        np.savetxt(self.behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
        np.savetxt(self.final_states_buffer, final_state.reshape(1, -1), delimiter=',')
        np.savetxt(self.cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
        # logs
        print(f'episode_reward: {performance}, oracle: {float(is_faulty)}, cell_selected_index: -1, cell_updated_index: {self.last_cell_updated}, nb_cells: {len(self.cells)}, execution_time: {exec_time}', file=self.logs_buffer)


    def record_mutated_execution(self, input: np.ndarray, performance: float, is_faulty: bool, feature: np.ndarray, final_state: np.ndarray, exec_time: float, parent: np.ndarray):
        # updates the collection structure
        behavior, cell = self.update_cell(input, performance, is_faulty, feature)
        # saves numpy arrays
        np.savetxt(self.inputs_buffer, input.reshape(1, -1), fmt='%1.0f', delimiter=',')
        np.savetxt(self.behaviors_buffer, behavior.reshape(1, -1), delimiter=',')
        np.savetxt(self.final_states_buffer, final_state.reshape(1, -1), delimiter=',')
        np.savetxt(self.cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
        # logs
        parent_index = self.get_cell(parent)
        print(f'episode_reward: {performance}, oracle: {float(is_faulty)}, cell_selected_index: {parent_index}, cell_updated_index: {self.last_cell_updated}, nb_cells: {len(self.cells)}, execution_time: {exec_time}', file=self.logs_buffer)


    def update_config(self, config: Dict):
        self.config.update(config)


class fuzzing:
    def __init__(self, rand_seed: int, sim_steps: int, **kwargs):
        self.corpus = []
        self.rewards = []
        self.result = []
        self.entropy = []
        self.coverage = []
        self.original = []
        self.count = []

        self.current_pose = None
        self.current_reward = None
        self.current_entropy = None
        self.current_coverage = None
        self.current_original = None
        self.current_index = None

        self.GMM = None
        self.GMMupdate = None
        self.GMMK = 10

        self.GMM_cond = None
        self.GMMupdate_cond = None
        self.GMMK_cond = 10
        self.GMMthreshold = 0.1
        # the previous paramaters come from Bipedal Walker use-case...

        #TODO
        self.rand_seed = rand_seed
        self.rng: np.random.Generator = np.random.default_rng(rand_seed)
        self.multivar_normal = multivariate_normal
        # last possible issue: giving the multivarial normal distribution object another (distinct) np.random.default_rng(rand_seed)
        self.multivar_normal.random_state = np.random.default_rng(rand_seed)
        # TODO: Framework adaptation (to gather results)
        self.version = 'MDPFuzz'
        self.creation_time = time.time()
        self.sim_steps = sim_steps

        self.test_budget = None
        self.init_budget = None

        self.config = {
            'rand_seed': self.rand_seed
        }

        # kwargs (name to include in the experimental configuration etc.)
        self.name = kwargs.get('name')
        if self.name is not None:
            self.config['name'] = self.name
        else:
            self.config['name'] = self.version


    def get_pose(self):
        choose_index = self.rng.choice(range(len(self.corpus)), 1, p=self.entropy / np.array(self.entropy).sum())[0]
        self.count[choose_index] -= 1
        self.current_index = choose_index
        self.current_pose = self.corpus[choose_index]
        self.current_reward = self.rewards[choose_index]
        self.current_entropy = self.entropy[choose_index]
        self.current_coverage = self.coverage[choose_index]
        self.current_original = self.original[choose_index]
        if self.count[choose_index] <= 0:
            self.corpus.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.coverage.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)
            self.current_index = None

        return self.current_pose


    def add_crash(self, result_pose):
        self.result.append(result_pose)
        choose_index = self.current_index
        if self.current_index != None:
            self.corpus.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.coverage.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)
            self.current_index = None


    def further_mutation(self, current_pose, rewards, entropy, cvg, original):
        choose_index = self.current_index
        copy_pose = copy.deepcopy(current_pose)

        if choose_index != None:
            self.corpus[choose_index] = copy_pose
            self.rewards[choose_index] = rewards
            self.entropy[choose_index] = entropy
            self.coverage[choose_index] = cvg
            self.count[choose_index] = 5
        else:
            self.corpus.append(copy_pose)
            self.rewards.append(rewards)
            self.entropy.append(entropy)
            self.coverage.append(cvg)
            self.original.append(original)
            self.count.append(5)


    def mutate(self, input: np.ndarray) -> np.ndarray:
        mutation = self.rng.choice(2, 15, p=[0.9, 0.1])
        if np.sum(mutation) == 0:
            mutation[0] = 1
        mutated_input = input + mutation
        mutated_input = np.remainder(mutated_input, 4)
        mutated_input = np.clip(mutated_input, 1, 3)
        return mutated_input


    def drop_current(self):
        choose_index = self.current_index
        if self.current_index != None:
            self.corpus.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.coverage.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)
            self.current_index = None


    def flatten_states(self, states):
        states = np.array(states)
        states_cond = np.zeros((states.shape[0]-1, states.shape[1] * 2))
        for i in range(states.shape[0]-1):
            states_cond[i] = np.hstack((states[i], states[i + 1]))
        return states, states_cond


    def GMMinit(self, data_corpus, data_corpus_cond):
        res = []
        for i in range(self.GMMK):
            temp = dict()
            temp[0] = 1 / self.GMMK
            temp[1] = temp[0] * np.mean(data_corpus[i:i+15], axis=0)
            temp[2] = np.zeros((data_corpus.shape[1], data_corpus.shape[1]))
            for j in range(i, i+15):
                temp[2] += temp[0] * np.matmul(data_corpus[j: j+1].T, data_corpus[j: j+1])
            temp[2] /= 15
            res.append(temp)

        weights = np.zeros(self.GMMK)
        means = np.zeros((self.GMMK, data_corpus.shape[1]))
        covariances = np.zeros((self.GMMK, data_corpus.shape[1], data_corpus.shape[1]))
        for i in range(self.GMMK):
            weights[i] = res[i][0]
            means[i] = res[i][1] / res[i][0]
            covariances[i] = np.eye(data_corpus.shape[1])

        self.GMM = dict()
        self.GMM['means'] = copy.deepcopy(means)
        self.GMM['weights'] = copy.deepcopy(weights)
        self.GMM['covariances'] = copy.deepcopy(covariances)

        res_cond = []
        for i in range(self.GMMK_cond):
            temp_cond = dict()
            temp_cond[0] = 1 / self.GMMK_cond
            temp_cond[1] = temp_cond[0] * np.mean(data_corpus_cond[i:i+15], axis=0)
            temp_cond[2] = np.zeros((data_corpus_cond.shape[1], data_corpus_cond.shape[1]))
            for j in range(i, i+15):
                temp_cond[2] += temp_cond[0] * np.matmul(data_corpus_cond[j: j+1].T, data_corpus_cond[j: j+1])
            temp_cond[2] /= 15
            res_cond.append(temp_cond)

        weights_cond = np.zeros(self.GMMK_cond)
        means_cond = np.zeros((self.GMMK_cond, data_corpus_cond.shape[1]))
        covariances_cond = np.zeros((self.GMMK_cond, data_corpus_cond.shape[1], data_corpus_cond.shape[1]))
        for i in range(self.GMMK_cond):
            weights_cond[i] = res_cond[i][0]
            means_cond[i] = res_cond[i][1] / res_cond[i][0]
            covariances_cond[i] = np.eye(data_corpus_cond.shape[1])

        self.GMM_cond = dict()
        self.GMM_cond['means'] = copy.deepcopy(means_cond)
        self.GMM_cond['weights'] = copy.deepcopy(weights_cond)
        self.GMM_cond['covariances'] = copy.deepcopy(covariances_cond)

        return res, res_cond


    def get_mdp_pdf(self, states_seq, states_seq_cond):
        first_frame = states_seq[0:1]
        GMMpdf = np.zeros(self.GMMK)
        for k in range(self.GMMK):
            GMMpdf[k] = self.GMM['weights'][k] * self.multivar_normal.pdf(first_frame, self.GMM['means'][k], self.GMM['covariances'][k])
        GMMpdf += 1e-5
        GMMpdfvalue = np.sum(GMMpdf)
        first_frame_pdf = GMMpdf

        single_frame_pdf = np.zeros((states_seq.shape[0], self.GMMK))
        other_frame_pdf = np.zeros((states_seq_cond.shape[0], self.GMMK_cond))

        for i in range(states_seq.shape[0]):
            for k in range(self.GMMK):
                single_frame_pdf[i, k] = self.GMM['weights'][k] * self.multivar_normal.pdf(states_seq[i], self.GMM['means'][k], self.GMM['covariances'][k])
        single_frame_pdf += 1e-5

        for i in range(states_seq_cond.shape[0]):
            for k in range(self.GMMK_cond):
                other_frame_pdf[i, k] = self.GMM_cond['weights'][k] * self.multivar_normal.pdf(states_seq_cond[i], self.GMM_cond['means'][k], self.GMM_cond['covariances'][k])
            other_frame_pdf[i] += 1e-5
            GMMpdfvalue *= np.min([np.sum(other_frame_pdf[i]) / np.sum(single_frame_pdf[i]), 1.0])
        return GMMpdfvalue, GMMpdf, other_frame_pdf


    def state_coverage(self, states_seq):
        states_seq, states_seq_cond = self.flatten_states(states_seq)
        if self.GMM == None:
            GMMresult, GMMresult_cond = self.GMMinit(states_seq, states_seq_cond)
            self.GMMupdate = dict()
            self.GMMupdate_cond = dict()
            self.GMMupdate['iter'] = 10
            self.GMMupdate['threshold'] = 0.05
            self.GMMupdate['S'] = copy.deepcopy(GMMresult)
            self.GMMupdate_cond['S'] = copy.deepcopy(GMMresult_cond)

        GMMpdfvalue, GMMpdf, other_frame_pdf = self.get_mdp_pdf(states_seq, states_seq_cond)
        first_frame = states_seq[0:1, :]

        if GMMpdfvalue < self.GMMthreshold:
            gamma = 1.0 / (self.GMMupdate['iter'])
            GMMpdf /= np.sum(GMMpdf)
            new_S = copy.deepcopy(self.GMMupdate['S'])

            for i in range(self.GMMK):
                new_S[i][0] = self.GMMupdate['S'][i][0] + gamma * (GMMpdf[i] - self.GMMupdate['S'][i][0])
                new_S[i][1] = self.GMMupdate['S'][i][1] + gamma * (GMMpdf[i]*first_frame - self.GMMupdate['S'][i][1])
                new_S[i][2] = self.GMMupdate['S'][i][2] + gamma * (GMMpdf[i]*np.matmul(first_frame.T, first_frame) - self.GMMupdate['S'][i][2])

            self.GMMupdate['S'] = copy.deepcopy(new_S)

            for i in range(self.GMMK):
                self.GMM['weights'][i] = new_S[i][0]
                self.GMM['means'][i] = new_S[i][1] / new_S[i][0]
                self.GMM['covariances'][i] = (new_S[i][2] - np.matmul(self.GMM['means'][i].reshape(1, -1).T, new_S[i][1])) / new_S[i][0]
                W, V = np.linalg.eigh(self.GMM['covariances'][i])
                W = np.maximum(W, 1e-3)
                D = np.diag(W)
                reconstruction = np.matmul(np.matmul(V, D), np.linalg.inv(V))
                self.GMM['covariances'][i] = copy.deepcopy(reconstruction)

            cond_choices = np.argsort(np.sum(other_frame_pdf, axis=1))
            for cond_index in cond_choices[:cond_choices.shape[0] // 10]:
                GMMpdf_cond = other_frame_pdf[cond_index]
                GMMpdf_cond /= np.sum(GMMpdf_cond)
                current_frame = states_seq_cond[cond_index: cond_index + 1, :]
                new_S_cond = copy.deepcopy(self.GMMupdate_cond['S'])

                for i in range(self.GMMK_cond):
                    new_S_cond[i][0] = self.GMMupdate_cond['S'][i][0] + gamma * (GMMpdf_cond[i] - self.GMMupdate_cond['S'][i][0])
                    new_S_cond[i][1] = self.GMMupdate_cond['S'][i][1] + gamma * (GMMpdf_cond[i]*current_frame - self.GMMupdate_cond['S'][i][1])
                    new_S_cond[i][2] = self.GMMupdate_cond['S'][i][2] + gamma * (GMMpdf_cond[i]*np.matmul(current_frame.T, current_frame) - self.GMMupdate_cond['S'][i][2])

                self.GMMupdate_cond['S'] = copy.deepcopy(new_S_cond)

                for i in range(self.GMMK_cond):
                    self.GMM_cond['weights'][i] = new_S_cond[i][0]
                    self.GMM_cond['means'][i] = new_S_cond[i][1] / new_S_cond[i][0]
                    self.GMM_cond['covariances'][i] = (new_S_cond[i][2] - np.matmul(self.GMM_cond['means'][i].reshape(1, -1).T, new_S_cond[i][1])) / new_S_cond[i][0]
                    W, V = np.linalg.eigh(self.GMM_cond['covariances'][i])
                    W = np.maximum(W, 1e-3)
                    D = np.diag(W)
                    reconstruction = np.matmul(np.matmul(V, D), np.linalg.inv(V))
                    self.GMM_cond['covariances'][i] = copy.deepcopy(reconstruction)

        return GMMpdfvalue


    def test_policy(self, model: BaseAlgorithm,
                    env_seed: int,
                    test_budget: int,
                    init_budget: int,
                    results_fp: str,
                    disable_pbar: bool = False):
        assert test_budget > init_budget
        self.test_budget = test_budget
        self.config['test_budget'] = self.test_budget
        self.init_budget = init_budget
        self.config['init_budget'] = self.init_budget

        self.config['env_seed'] = env_seed

        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp

        # directly loads the edges for the use-case
        feature_edges = np.load(f'grid/bw/{env_seed}_{self.sim_steps}_edges.npy')
        collections: List[GridCollection] = []

        for expert_indices in EXPERT_INDICES:
            collections.append(GridCollection(f'{filepath}{expert_indices[0]}{expert_indices[1]}', expert_indices, feature_edges, copy.deepcopy(self.config)))

        # the search only manages the original MDPFuzz's logs
        mdpfuzz_logs_buffer = open(f'{filepath}_mdpfuzz_logs.txt', 'w', buffering=1)
        testing_start_time = time.time()

        # accumulators for the collection initialization
        inputs: List[np.ndarray] = []
        features: List[np.ndarray] = []
        final_states: List[np.ndarray] = []
        acc_rewards: List[float] = []
        oracles: List[bool] = []
        execution_times: List[float] = []

        # initialization of MDPFuzz
        i = 0
        nb_iterations = 0
        pbar = tqdm.tqdm(total=init_budget)

        while i < init_budget:
            input: np.ndarray = self.rng.integers(low=1, high=4, size=15)

            episode_reward, done, feature, obs_seq, exec_time = execute_policy_trajectory(input, model, env_seed=env_seed, sim_steps=self.sim_steps)

            execution_times.append(exec_time)
            inputs.append(input)
            features.append(feature)
            acc_rewards.append(episode_reward)
            oracles.append(done)
            final_states.append(obs_seq[-1])

            if not done:
                mutate_states = self.mutate(input)
                episode_reward_mutate, done_mutate, feature_mutate, final_state_mutate, exec_time_mutate = execute_policy(mutate_states, model, env_seed=env_seed, sim_steps=self.sim_steps)

                entropy = np.abs(episode_reward_mutate - episode_reward)
                cvg = self.state_coverage(obs_seq)
                self.further_mutation(input, episode_reward, entropy, cvg, input)
                print(f'sensitivity: {entropy}, episode_reward: {episode_reward}, episode_reward_mutate: {episode_reward_mutate}, oracle: {float(done_mutate)}, coverage: {cvg}', file=mdpfuzz_logs_buffer)

                inputs.append(mutate_states)
                features.append(feature_mutate)
                acc_rewards.append(episode_reward_mutate)
                oracles.append(done_mutate)
                execution_times.append(exec_time_mutate)
                final_states.append(final_state_mutate)

                i += 1
                nb_iterations += 1
                pbar.update(1)

            nb_iterations += 1
        pbar.close()

        # logs the execution results and fills the collection after the initialization
        for j in range(nb_iterations):
            for collection in collections:
                collection.record_init_execution(inputs[j], acc_rewards[j], oracles[j], features[j], final_states[j], execution_times[j])

        self.config['init_iterations'] = nb_iterations

        executions_budget = test_budget - nb_iterations if test_budget > 12 else 10000
        time_budget = min(12, executions_budget) * 3600
        pbar = tqdm.tqdm(total=executions_budget)
        print(f'Time budget of {(time_budget / 60):.2f} minutes; bound to {executions_budget} executions.')

        cvg_threshold = 0.1 # follows the paper and not the author's experiments replication repository
        start_time = time.time()
        current_time = time.time()
        nb_executions = 0
        pbar = tqdm.tqdm(total=executions_budget, disable=disable_pbar)

        while (current_time - start_time < time_budget) and (nb_executions < executions_budget) and (len(self.corpus) > 0):
            states = self.get_pose()
            mutate_states = self.mutate(states)

            episode_reward, done, feature, sequences, exec_time = execute_policy_trajectory(mutate_states, model, env_seed=env_seed, sim_steps=self.sim_steps)
            execution_times.append(exec_time)

            cvg = self.state_coverage(sequences)
            local_sensitivity = np.abs(episode_reward - self.current_reward)
            if done:
                self.add_crash(mutate_states)
            if cvg < cvg_threshold or episode_reward < self.current_reward:
                    current_pose = copy.deepcopy(mutate_states)
                    orig_pose = self.current_original
                    self.further_mutation(current_pose, episode_reward, local_sensitivity, cvg, orig_pose)

            # keeps logging original information
            print(f'sensitivity: {local_sensitivity}, episode_reward: {self.current_reward}, episode_reward_mutate: {episode_reward}, oracle: {float(done)}, coverage: {cvg}', file=mdpfuzz_logs_buffer)
            # updates all collections
            for collection in collections:
                collection.record_mutated_execution(mutate_states, episode_reward, done, feature, sequences[-1], exec_time, states)
            current_time = time.time()
            nb_executions += 1
            pbar.update(1)


        testing_end_time = time.time()
        self.config['testing_start_time'] = testing_start_time
        self.config['testing_end_time'] = testing_end_time
        self.config['testing_time'] = testing_end_time - testing_start_time
        self.config['total_execution_time'] = sum(execution_times)

        pbar.close()
        mdpfuzz_logs_buffer.close()

        for collection in collections:
            collection.update_config(self.config)
            collection.save_state()


    def random_testing(self, model: BaseAlgorithm,
                    env_seed: int,
                    test_budget: int,
                    results_fp: str,
                    disable_pbar: bool = False):
        '''Random testing loop baseline.'''
        self.test_budget = test_budget
        self.config['test_budget'] = self.test_budget
        self.config['env_seed'] = env_seed


        if os.path.isdir(results_fp):
            filepath = f'{results_fp}{self.creation_time}' if results_fp.endswith('/') else f'{results_fp}/{self.creation_time}'
        else:
            filepath = results_fp

        feature_edges = np.load(f'grid/bw/{env_seed}_{self.sim_steps}_edges.npy')
        collections: List[GridCollection] = []

        for expert_indices in EXPERT_INDICES:
            collections.append(GridCollection(f'{filepath}{expert_indices[0]}{expert_indices[1]}', expert_indices, feature_edges, copy.deepcopy(self.config)))

        time_budget = min(12, test_budget) * 3600
        executions_budget = test_budget if test_budget > 12 else 10000
        print(f'Time budget of {(time_budget / 60):.2f} minutes; bound to {executions_budget} executions.')

        execution_times = []

        start_time = time.time()
        current_time = time.time()
        nb_executions = 0
        pbar = tqdm.tqdm(total=executions_budget, disable=disable_pbar)

        while (current_time - start_time < time_budget) and (nb_executions < executions_budget):
            input: np.ndarray = self.rng.integers(low=1, high=4, size=15)
            episode_reward, oracle, feature, fs, exec_time = execute_policy(input, model, env_seed=env_seed, sim_steps=self.sim_steps)
            execution_times.append(exec_time)

            for collection in collections:
                collection.record_init_execution(input, episode_reward, oracle, feature, fs, exec_time)

            current_time = time.time()
            nb_executions += 1
            pbar.update(1)

        testing_end_time = time.time()
        self.config['testing_start_time'] = start_time
        self.config['testing_end_time'] = testing_end_time
        self.config['testing_time'] = testing_end_time - start_time
        self.config['total_execution_time'] = sum(execution_times)

        pbar.close()

        for collection in collections:
            collection.update_config(self.config)
            collection.save_state()


if __name__ == '__main__':
    torch.set_num_threads(1)
    main_seed = 2021
    env_seed = 0
    model = load_model()

    test_budget = 5000
    init_budget = 1000


    results_folder = 'results/bwrt/'
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    for seed in EXPERIMENT_SEEDS:
        fuzzer = fuzzing(seed, 300, name='Random Testing')
        fuzzer.random_testing(model, env_seed, test_budget, results_folder)

    results_folder = 'results/bw_mdpfuzz/'
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    for seed in EXPERIMENT_SEEDS:
        fuzzer = fuzzing(seed, 300)
        fuzzer.test_policy(model, env_seed, test_budget, init_budget, results_folder)