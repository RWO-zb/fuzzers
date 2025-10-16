import json
import copy
import tqdm
import torch
import os
import time
import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from typing import List
from stable_baselines3.common.base_class import BaseAlgorithm

from result_analysis import isin
from common import EXPERIMENT_SEEDS
import tt_common as tt

class fuzzing:
    def __init__(self, rand_seed: int, **kwargs):
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
        self.multivar_normal.random_state = np.random.default_rng(rand_seed)
        # TODO: Framework adaptation (to gather results)
        self.version = 'MDPFuzz'
        self.creation_time = time.time()

        self.test_budget = None
        self.init_budget = None

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


    def mutate(self, states: np.ndarray):
        return tt.mutate(states, self.rng)


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


    #TODO
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
        cell_dfs = []
        for i, cell_data in enumerate(self.cells_data):
            cell_dfs.append(
                pd.DataFrame.from_records(
                    data=[[score, is_faulty, i] + self.cells[i] + behavior.tolist() for (_input, score, is_faulty, behavior) in cell_data],
                    columns=['score', 'is_faulty', 'cell_index'] + [f'cell{i}' for i in range(2)] + [f'behavior{i}' for i in range(2)]
                    )
                )
        pd.concat(cell_dfs, ignore_index=True).to_csv(f'{filepath}_data.csv', index=0)
        np.save(f'{filepath}_inputs.npy', np.concatenate([np.array(list(map(lambda x: x[0], cell_data))) for cell_data in self.cells_data]))
        self.save_random_state(filepath)
        self.save_configuration(filepath)


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


    def get_cell(self, input: np.ndarray):
        '''Finds the input's data in the collection and return its cell index.'''
        for i in range(len(self.cells_data)):
            inputs = np.array(list(map(lambda x: x[0], self.cells_data[i])))
            if isin(inputs, input).any():
                return i
        # should never happen
        raise ValueError(f'Input has not been found: {input}')


    def test_policy(self, model: BaseAlgorithm,
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

        mdpfuzz_logs_buffer = open(f'{filepath}_mdpfuzz_logs.txt', 'w', buffering=1) # original logs

        inputs: List[np.ndarray] = []
        features: List[np.ndarray] = []
        final_states: List[int] = []
        acc_rewards: List[float] = []
        oracles: List[bool] = []
        execution_times = []
        testing_start_time = time.time()

        # behavior space
        bs = tt.BehaviorSpace()

        # initialization of MDPFuzz
        i = 0
        nb_iterations = 0
        pbar = tqdm.tqdm(total=init_budget)

        # ensures solutions are always new
        evaluated_solutions = []

        while i < init_budget:
            input: np.ndarray = tt.generate_input(self.rng)
            if not (input.tolist() in evaluated_solutions):
                evaluated_solutions.append(input.tolist())
            t0 = time.time()
            episode_reward, done, feature, obs_seq, _ = tt.execute_policy_trajectory(model, input, self._env)
            state_seq = np.vstack([list(self._env.decode(obs)) for obs in obs_seq])
            t1 = time.time()
            execution_times.append(t1 - t0)

            inputs.append(input)
            features.append(feature)
            acc_rewards.append(episode_reward)
            oracles.append(done)
            final_states.append(obs_seq[-1])

            if not done:
                # +/- fixes mutation operator (but mutants still might be already evaluated...)
                attempts = 1
                while attempts < 40:
                    mutate_states = self.mutate(input)
                    tmp = mutate_states.tolist()
                    if not (tmp in evaluated_solutions):
                        evaluated_solutions.append(tmp)
                        break
                    else:
                        attempts += 1

                t0 = time.time()
                episode_reward_mutate, done_mutate, feature_mutate, final_state_mutate, _ = tt.execute_policy(model, mutate_states, self._env)
                t1 = time.time()
                execution_times.append(t1 - t0)

                entropy = np.abs(episode_reward_mutate - episode_reward)
                cvg = self.state_coverage(state_seq)
                self.further_mutation(input, episode_reward, entropy, cvg, input)
                print(f'evaluated_solutions: {len(evaluated_solutions)}, sensitivity: {entropy}, episode_reward: {episode_reward}, episode_reward_mutate: {episode_reward_mutate}, oracle: {float(done_mutate)}, coverage: {cvg}', file=mdpfuzz_logs_buffer)

                inputs.append(mutate_states)
                features.append(feature_mutate)
                acc_rewards.append(episode_reward_mutate)
                oracles.append(done_mutate)
                final_states.append(final_state_mutate)

                i += 1
                nb_iterations += 1
                pbar.update(1)

            nb_iterations += 1

        pbar.close()

        inputs_buffer = open(f'{filepath}_inputs.txt', 'w', buffering=1)
        final_states_buffer = open(f'{filepath}_final_states.txt', 'w', buffering=1)
        cells_buffer = open(f'{filepath}_cells.txt', 'w', buffering=1)
        logs_buffer = open(f'{filepath}_logs.txt', 'w', buffering=1)
        # records both the features and the (likely to be the 2d) behaviors
        behaviors_buffer = open(f'{filepath}_behaviors.txt', 'w', buffering=1)
        features_buffer = open(f'{filepath}_features.txt', 'w', buffering=1)

        # logs the execution results and fills the collection after the initialization
        for j in range(nb_iterations):
            behavior, cell = bs.describe(features[j])
            updated_cell_index = self.update_cell(cell, inputs[j], acc_rewards[j], oracles[j], behavior)

            print(f'episode_reward: {acc_rewards[j]}, oracle: {float(oracles[j])}, cell_selected_index: -1, cell_updated_index: {updated_cell_index}, nb_cells: {len(self.cells)}, execution_time: {execution_times[j]}', file=logs_buffer)
            # the data is composed of integers
            np.savetxt(inputs_buffer, inputs[j].reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(features_buffer, feature.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
            final_states_buffer.write(f'{final_states[j]}\n')


        self.config['init_iterations'] = nb_iterations
        # the remaining test budget accounts for the total cost of the initialization phase of the fuzzer (that discards faults)
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
            # +/- fixes mutation operator (but mutants still might be already evaluated...)
            attempts = 1
            while attempts < 40:
                mutate_states = self.mutate(states)
                tmp = mutate_states.tolist()
                if not (tmp in evaluated_solutions):
                    evaluated_solutions.append(tmp)
                    break
                else:
                    attempts += 1

            t0 = time.time()
            episode_reward, done, feature, obs_seq, _ = tt.execute_policy_trajectory(model, mutate_states, self._env)
            state_seq = np.vstack([list(self._env.decode(obs)) for obs in obs_seq])
            t1 = time.time()
            execution_times.append(t1 - t0)

            cvg = self.state_coverage(state_seq)
            local_sensitivity = np.abs(episode_reward - self.current_reward)
            if done:
                self.add_crash(mutate_states)
            if cvg < cvg_threshold or episode_reward < self.current_reward:
                    current_pose = copy.deepcopy(mutate_states)
                    orig_pose = self.current_original
                    self.further_mutation(current_pose, episode_reward, local_sensitivity, cvg, orig_pose)


            # keeps logging original information
            print(f'evaluated_solutions: {len(evaluated_solutions)}, sensitivity: {local_sensitivity}, episode_reward: {self.current_reward}, episode_reward_mutate: {episode_reward}, oracle: {float(done)}, coverage: {cvg}', file=mdpfuzz_logs_buffer)

            # gets the cell of the input selected by MDPFuzz
            cell_index = self.get_cell(states)
            # behavior and cell from the feature
            behavior, cell = bs.describe(feature)
            mutate_states_index = self.update_cell(cell, mutate_states, episode_reward, done, behavior)

            print(f'episode_reward: {episode_reward}, oracle: {float(done)}, cell_selected_index: {cell_index}, cell_updated_index: {mutate_states_index}, nb_cells: {len(self.cells)}, execution_time: {t1 - t0}', file=logs_buffer)
            np.savetxt(inputs_buffer, mutate_states.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(behaviors_buffer, behavior.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(features_buffer, feature.reshape(1, -1), fmt='%1.0f', delimiter=',')
            np.savetxt(cells_buffer, np.array(cell).reshape(1, -1), fmt='%1.0f', delimiter=',')
            final_states_buffer.write(f'{obs_seq[-1]}\n')

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
        behaviors_buffer.close()
        features_buffer.close()
        final_states_buffer.close()
        inputs_buffer.close()
        cells_buffer.close()
        logs_buffer.close()
        self.save_state(filepath)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main_seed = 2021
    env_seed = 0
    model = tt.load_taxi_model()

    test_budget = 5000
    init_budget = 1000

    results_fp = f'results/tt_mdpfuzz'
    if not os.path.isdir(results_fp):
        os.mkdir(results_fp)

    for seed in EXPERIMENT_SEEDS:
        fuzzer = fuzzing(seed)
        fuzzer.test_policy(model, test_budget, init_budget, results_fp)