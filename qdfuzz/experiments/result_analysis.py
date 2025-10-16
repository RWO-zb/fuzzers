import os
import torch
import json
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from typing import List, Union, Dict, Tuple, Iterable
from sklearn.neighbors import NearestNeighbors

from common import read_results_from_folder

FAULT_LABEL = '#Faults'
BS_LABEL = '#BS'
FBS_LABEL = '#FBS'
KNN_LABEL = 'Final State Diversity'
FKNN_LABEL ='Failure State Diversity'
AXIS_LABEL_FONTSIZE = 15
TITLE_LABEL_FONTSIZE = 16


#################################################################################################################################
############################################################## HELPERS ##########################################################


def filter_data(boolean_list: List[np.ndarray], points_list: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Accumulates the number of points which are both unique and filtered by a given boolean mask.

    For instance, given the mask [0 0 1 1 0 0 1] and the points [1 2 3 3 4 5 5]:
        - The merged mask of the relevant points is [0 0 1 0 0 0 1].
        - Therefore, the result of their accumulation is [0 0 1 1 1 1 2].

    It returns a list of tuples as:
    1. Unique filtered points
    2. Their indices in the original points
    3. The corresponding mask and
    4. The accumulation of the latter

    Therefore, 3 and 4 have the same length of the points while the 1 and 2 share a length which is lower or equal to the length of the points.
    '''
    n = len(boolean_list)
    if points_list is not None:
        assert len(points_list) == n
        for bl, pl in zip(boolean_list, points_list):
            assert len(bl) == len(pl)


    def is_in_list(list: List[np.ndarray], element: np.ndarray):
        for e in list:
            if np.array_equal(e, element):
                return True
        return False


    results = []
    for oracle_mask, points in zip(boolean_list, points_list):
        if len(points.shape) == 1:
            points = points[:, None]
        data = []
        data_indices = []
        data_mask = []
        accumulator = []
        for i in range(len(oracle_mask)):
            oracle, point = oracle_mask[i], points[i]
            # only updates the result lists if unseen faulty
            if oracle and (not is_in_list(data, point)):
                data.append(point)
                data_indices.append(i)
                data_mask.append(oracle) # True
            else:
                data_mask.append(False)

        # if I handle i == 0 before, I can also accumulate during the previous for loop...
        counter = 0
        for b in data_mask:
            counter += int(b)
            accumulator.append(counter)

        data = np.array(data)
        data_indices = np.array(data_indices)
        data_mask = np.array(data_mask)
        accumulator = np.array(accumulator)

        # print('oracle mask:', oracle_mask)
        # print('mask to accumulate:', data_mask)
        # print('indices in the points:', data_indices)
        # print('-----------------------------------------------------------')
        results.append((data, data_indices, data_mask, accumulator))

    return results


def knn_dists(data: np.ndarray, k: int) -> np.ndarray:
    '''
    Distance to k nearest neighbours.
    This is the sparseness criterion of the original novelty search paper.
    Intuitively, if the average distance to a given point's nearest
    neighbors is large then it is in a sparse area; it is in a dense region if the average
    distance is small.
    It returns all the mean distances.
    '''
    if not isinstance(data, np.ndarray):
        return np.nan
    u_data = np.unique(data, axis=0)
    # nb_totals = data.shape[0]
    # nb_uniques = u_data.shape[0]
    # print(f'{nb_uniques} out of {nb_totals} points are unique')
    if len(u_data.shape) == 1:
        u_data = u_data.reshape(-1, 1)
    neighbors = NearestNeighbors(n_neighbors=k).fit(u_data)
    distances, _ = neighbors.kneighbors()
    return np.mean(distances, axis=1)


#####################################################################################
################################### Novelty Search data analysis ####################


def ns_analysis(ns_res: List[Dict]):
    for res in ns_res:
        # adds the popsize and nov_threshold parameters to the name
        popsize = res['config']['pop_size']
        t = res['config']['nov_threshold']
        ns_logs_df: pd.DataFrame = res['ns_logs']

        res['config']['name'] += f' popsize/threshold = ({popsize}, {t})'
        res['archive_sizes'] = [s for s in ns_logs_df['archive_size'].to_numpy() for _ in range(popsize)]
        res['archive_sparsenesses'] = [s for s in ns_logs_df['archive_sparseness'].to_numpy() for _ in range(popsize)]
    return ns_res


def plot_ns_analysis(ns_res: List[Dict]):
        def compute_statistics(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            '''Helper that computes statistical results from a set of results.'''
            if not isinstance(data, np.ndarray):
                data: np.ndarray = np.array(data)
            x = np.arange(data.shape[1])
            y = np.median(data, axis=0)
            perc_25 = np.percentile(data, 25, axis=0)
            perc_75 = np.percentile(data, 75, axis=0)
            return y, perc_25, perc_75, x


        use_cases: List[str] = np.unique([d['config']['use_case'] for d in ns_res]).tolist()
        nb_use_cases = len(use_cases)
        methods_names: List[str] = np.unique([d['config']['name'] for d in ns_res]).tolist()
        cmap = plt.cm.jet # plt.cm.jet is a LinearSegmentedColormap
        rgba_colors = [cmap(i) for i in np.linspace(0, 1, len(methods_names))]
        colors_dict = {n: c for n, c in zip(methods_names, rgba_colors)}

        fig, axs = plt.subplots(nrows=nb_use_cases, ncols=2, figsize=(15,7*nb_use_cases))
        if nb_use_cases == 1:
            axs = [axs]

        axs[0][0].set_title('Archive sizes')
        axs[0][1].set_title('(distinct) #Faults')
        for u in range(nb_use_cases):
            axs[u][0].set_ylabel(use_cases[u])
            axs[u][0].set_xlabel('#Iterations')
            axs[u][1].set_xlabel('#Iterations')

            for method_name in methods_names:
                sub_results = [d for d in ns_res if (d['config']['use_case'] == use_cases[u]) and (d['config']['name'] == method_name)]
                if len(sub_results) == 0:
                    print(f'No result found for use-case {use_cases[u]} and methodology {method_name}')
                    continue
                color = colors_dict[method_name]
                label = method_name

                sizes = [d['archive_sizes'] for d in sub_results]
                y, perc_25, perc_75, x = compute_statistics(sizes)
                axs[u][0].plot(x, y, color=color, label=label)
                axs[u][0].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)

                inputs = [d['inputs'] for d in sub_results]
                oracles = [d['logs']['oracle'].to_numpy() for d in sub_results]
                nb_faults = compute_evolution_fault_triggering_inputs(inputs, oracles)
                y, perc_25, perc_75, x = compute_statistics(nb_faults)
                color = colors_dict[method_name]
                label = method_name
                axs[u][1].plot(x, y, color=color, label=label)
                axs[u][1].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
        legend = axs[-1][0].legend(title='Testing Methodology')
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor('0.9')
        legend_frame.set_edgecolor('0.9')

        fig.tight_layout()
        return (fig, axs)


##############################################################################################
#################################### Plotting parameters #####################################


def color_data(data: List[Dict]) -> Tuple[List[str], List[str], Dict[str, Tuple[float]]]:
    use_cases: List[str] = np.unique([d['config']['use_case'] for d in data]).tolist()
    methods_names: List[str] = np.unique([d['config']['name'] for d in data]).tolist()
    cmap = plt.cm.jet # plt.cm.jet is a LinearSegmentedColormap
    rgba_colors = [cmap(i) for i in np.linspace(0, 1, len(methods_names))]
    colors_dict = {n: c for n, c in zip(methods_names, rgba_colors)}
    return use_cases, methods_names, colors_dict


##############################################################################################
############ RQ1: How many (distinct) fault-triggering inputs do frameworks find? ############


def compute_evolution_fault_triggering_inputs(inputs: List[np.ndarray], oracles: List[np.ndarray]):
    '''
    Computes the evolution of the number of fault-triggering inputs found for each set of results.
    As such, the @inputs and @oracles lists re expected to be of the same length, every ith data belonging to a particular testing methodology.
    Redundant inputs are not considered.
    '''
    return [res[-1] for res in filter_data(oracles, inputs)]


def compute_rq1_results(data: List[Dict]) -> List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    use_cases: List[str] = np.unique([d['config']['use_case'] for d in data]).tolist()
    methods_names: List[str] = np.unique([d['config']['name'] for d in data]).tolist()

    def compute_statistics(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            data: np.ndarray = np.array(data)
        y = np.median(data, axis=0)
        perc_25 = np.percentile(data, 25, axis=0)
        perc_75 = np.percentile(data, 75, axis=0)
        return y, perc_25, perc_75

    results_list = []
    for case in use_cases:
        results = {}

        for method_name in methods_names:
            method_data = [d for d in data if (d['config']['use_case'] == case) and (d['config']['name'] == method_name)]

            if len(method_data) == 0:
                print(f'No result found for use-case {case} and methodology {method_name}')
                continue

            inputs = [d['inputs'] for d in method_data]
            oracles = [d['logs']['oracle'].to_numpy() for d in method_data]

            results[method_name] = compute_statistics(compute_evolution_fault_triggering_inputs(inputs, oracles))

        results_list.append(results)

    return results_list


def plot_rq1_results(
        use_cases: List[str],
        colors_dict: Dict[str, Tuple],
        results: List[Dict[str, Union[Tuple, np.ndarray]]]
    ):
    '''Plots the results for RQ1.'''

    n = len(use_cases)
    fig, axs = plt.subplots(ncols=n, figsize=(7*n, 6), sharex=True)
    if n == 1:
        axs = [axs]
        axs[0].grid(axis='y', color='0.9', linestyle='-', linewidth=1)
    else:
        [ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1) for ax in axs.flat]

    axs[0].set_ylabel(FAULT_LABEL, fontsize=AXIS_LABEL_FONTSIZE)

    for u in range(n):
        data = results[u]
        ax = axs[u]
        ax.set_xlabel('Iterations', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(use_cases[u], fontsize=TITLE_LABEL_FONTSIZE)
        for name, data in data.items():
            color = colors_dict[name]
            label = name
            if isinstance(data, np.ndarray):
                ax.plot(np.arange(len(data)), data, color=color, label=label)
            else:
                y, perc_25, perc_75 = data
                x = np.arange(len(y))
                ax.plot(x, y, color=color, label=label)
                ax.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)

    ax = axs[np.argmax([len(d.keys()) for d in results])]
    legend = ax.legend(prop={'size': 10}, labelspacing=1.1, handletextpad=1.05, borderpad=1.05, borderaxespad=1.0)
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('0.9')
    legend_frame.set_edgecolor('0.9')
    fig.tight_layout()
    return (fig, axs)


#############################################################################################################
################## RQ2: How well QD-based testing methodologies improve test coverage/diversity? ############


def compute_evolution_cells(data: List[pd.DataFrame]):
    '''Computes the number of cells discovered over search iterations.'''
    results = [df['nb_cells'].to_numpy() for df in data]
    return results


def compute_mean_knn(results: List[Dict], knn: int = 3, computation_steps: int = 50):
    '''Computes the knn means and returns all the data required to plot them.'''
    use_cases: List[str] = np.unique([d['config']['use_case'] for d in results]).tolist()
    methods_names: List[str] = np.unique([d['config']['name'] for d in results]).tolist()
    seeds: List[int] = np.unique([d['config']['rand_seed'] for d in results]).tolist()

    # print(use_cases)
    # print(methods_names)
    # print(seeds)

    # LinearSegmentedColormap
    cmap = plt.cm.jet
    rgba_colors = [cmap(i) for i in np.linspace(0, 1, len(methods_names))]
    # colors of each method name
    colors_dict = {n: c for n, c in zip(methods_names, rgba_colors)}

    #TODO: MAP-Elite's elites filtering later
    # add_elite_results = kwargs.get('add_elite_results', False)

    assert computation_steps > 0

    # re-aranges the results per use_case
    use_cases_dict: Dict[str, Dict[str, List]] = {}
    for case in use_cases:
        use_cases_dict[case] = {}
        for name in methods_names:
            # case's results
            sub_results = [d for d in results if (d['config']['use_case'].startswith(case)) and (d['config']['name'] == name)]
            # RT and MDPFuzz are redundant on Bipedal Walker case since they do not account for behaviors
            if case.startswith('Bipedal Walker') and (name in ['Random Testing', 'MDPFuzz']):
                non_redundant_results = []
                for seed in seeds:
                    tmp = [d for d in sub_results if d['config']['rand_seed'] == seed]
                    if len(tmp) > 0:
                        non_redundant_results.append(tmp[0])
                use_cases_dict[case].update({name: non_redundant_results})
            else:
                use_cases_dict[case].update({name: sub_results})
        # logs
        print(f'For {case}:')
        for k, v in use_cases_dict[case].items():
            print(k, len(v), np.unique([d['config']['rand_seed'] for d in v]).tolist())
        print('----------------------------------------------')


    def dist(arr: np.ndarray, knn: int) -> np.ndarray:
        # to handle integer data of Taxi
        if len(arr.shape) == 1:
            arr = arr.reshape(-1, 1)
        data = np.unique(arr, axis=0)
        neighbors = NearestNeighbors(n_neighbors=knn).fit(data)
        distances, _ = neighbors.kneighbors()
        return np.mean(distances, axis=1)


    def statistical_dists(arr_list: List[np.ndarray], knn: int, steps: Iterable = None):
        if steps is None:
            max_length = np.max([len(arr) for arr in arr_list])
            steps = range(computation_steps - 1, max_length, computation_steps)

        medians, q1s, q3s = [], [], []
        for arr in arr_list:
            dists_list = [dist(arr[:step], knn) for step in steps]
            medians.append(np.array([np.median(d) for d in dists_list]))
            q1s.append(np.array([np.quantile(d, 0.25) for d in dists_list]))
            q3s.append(np.array([np.quantile(d, 0.75) for d in dists_list]))
        return medians, q1s, q3s, steps


    def statistical_dists_splits(data_list: List[List[np.ndarray]], knn: int = 5) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        medians, q1s, q3s = [], [], []
        for data in data_list:
            mean_distances_list = [knn_dists(d, knn) for d in data]
            # uses nan version (knn_dists can return np.nan)
            medians.append(np.array([np.median(d) if isinstance(d, np.ndarray) else np.nan for d in mean_distances_list]))
            q1s.append(np.array([np.quantile(d, 0.25) if isinstance(d, np.ndarray) else np.nan for d in mean_distances_list]))
            q3s.append(np.array([np.quantile(d, 0.75) if isinstance(d, np.ndarray) else np.nan for d in mean_distances_list]))
        return medians, q1s, q3s

    # dict of (the means of) the medians and quantiles for all methodologies per use-case
    results_dicts: List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    # similar data but with distinct fault-triggering final states
    fresults_dicts: List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    # the first indices to plot the results above (since there is no guarantee to have such final state data in the first splits)
    x_starts_dicts = []
    for case in use_cases:
        # results of all methodologies
        method_result_dict = {}
        method_fresult_dict = {}
        x_starts_dict = {}
        for method_name, sub_results in use_cases_dict[case].items():
            if len(sub_results) == 0:
                print(f'No result found for use-case {case} and methodology {method_name}')
                continue

            fs_list = [d['final_states'] for d in sub_results]

            # final states
            medians, q1s, q3s, x = statistical_dists(fs_list, knn)
            medians = np.vstack(medians)
            q1s = np.vstack(q1s)
            q3s = np.vstack(q3s)
            # reduces the distribution of the data (medians and quantiles) their mean
            mmedians = np.mean(medians, axis=0)
            mq1s = np.mean(q1s, axis=0)
            mq3s = np.mean(q3s, axis=0)
            method_result_dict[method_name] = (mmedians, mq1s, mq3s)

            # distinct fault-triggering data filtering
            max_length = np.max([len(fs) for fs in fs_list])
            x = range(computation_steps - 1, max_length, computation_steps)
            x_starts = []
            final_states_splits = []
            oracles = [d['logs']['oracle'].to_numpy() for d in sub_results]
            for fs, oracle_mask in zip(fs_list, oracles):
                mask = filter_data([oracle_mask], [fs])[0][2]
                indices = np.ravel(np.argwhere(mask))
                # the start index is thus the index from which there are (nb neighbors + 1) faulty final states before
                start_index = indices[knn]
                # print(f'{case} {method_name}: enough faults from index {start_index}.')
                x_starts.append(start_index)
                split = [fs[:(step+1)][mask[:(step+1)]] if step > start_index else np.nan for step in x]
                # print(split)
                # tmp = [l.shape if isinstance(l, np.ndarray) else l for l in split]
                # print(tmp)
                final_states_splits.append(split)
            # print(method_name, x_starts)
            x_starts_dict[method_name] = np.min(x_starts)
            medians, q1s, q3s = statistical_dists_splits(final_states_splits, knn)
            mmedians = np.nanmean(medians, axis=0)
            mq1s = np.nanmean(q1s, axis=0)
            mq3s = np.nanmean(q3s, axis=0)
            method_fresult_dict[method_name] = (mmedians, mq1s, mq3s)

        results_dicts.append(method_result_dict)
        fresults_dicts.append(method_fresult_dict)
        x_starts_dicts.append(x_starts_dict)
    return use_cases, results_dicts, fresults_dicts, x_starts_dicts, colors_dict


def compute_relative_performance(results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], name_ref: str = 'Random Testing'):
    '''
    Returns the relative performance to the set of results labeled @name_ref.
    Even though it assumes statistical results (as medians and 1/3 quantiles), the relative performance is computed on the medians only.
    '''
    relative_results: Dict[str, np.ndarray] = {}
    if results.get(name_ref, None) is None:
        print(f'No result found for the reference {name_ref}.')
        return results
    m_ref, q1_ref, q3_ref = results[name_ref]
    for name, (m, q1, q3) in results.items():
        relative_results[name] = ((m - m_ref) / m_ref) + 1
    return relative_results


def compute_rq2_bs_results(data: List[Dict]):
    use_cases: List[str] = np.unique([d['config']['use_case'] for d in data]).tolist()
    methods_names: List[str] = np.unique([d['config']['name'] for d in data]).tolist()

    def compute_statistics(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            data: np.ndarray = np.array(data)
        y = np.median(data, axis=0)
        perc_25 = np.percentile(data, 25, axis=0)
        perc_75 = np.percentile(data, 75, axis=0)
        return y, perc_25, perc_75

    bs_coverages = []
    faulty_bs_coverages = []
    for case in use_cases:
        bs_cov = {}
        faulty_bs_cov = {}

        for method_name in methods_names:
            method_data = [d for d in data if (d['config']['use_case'] == case) and (d['config']['name'] == method_name)]

            if len(method_data) == 0:
                print(f'No result found for use-case {case} and methodology {method_name}')
                continue

            cells = [d['cells'] for d in method_data]
            dfs = [d['logs'] for d in method_data]
            oracles = [d['logs']['oracle'].to_numpy() for d in method_data]

            bs_cov[method_name] = compute_statistics(compute_evolution_cells(dfs))
            faulty_bs_cov[method_name] = compute_statistics([res[-1] for res in filter_data(oracles, cells)])

        bs_coverages.append(bs_cov)
        faulty_bs_coverages.append(faulty_bs_cov)

    return bs_coverages, faulty_bs_coverages


def plot_rq2_bs_results(
        use_cases: List[str],
        colors_dict: Dict[str, Tuple],
        bs_results: List[Dict[str, Tuple]],
        faulty_bs_results: List[Dict[str, np.ndarray]]
    ):
    '''Plots the behavior coverage results for RQ2.'''

    nb_use_cases = len(use_cases)

    fig, axs = plt.subplots(nrows=nb_use_cases, ncols=2, figsize=(10, 4*nb_use_cases), sharex=True)
    if nb_use_cases == 1:
        axs = [axs]
        [ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1) for ax in axs[0].flat]
    else:
        [ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1) for ax in axs.flat]

    axs[0][0].set_title('#Behaviours', fontsize=TITLE_LABEL_FONTSIZE)
    axs[0][1].set_title('#Faulty Behaviours', fontsize=AXIS_LABEL_FONTSIZE)
    axs[-1][0].set_xlabel('Iterations', fontsize=AXIS_LABEL_FONTSIZE)
    axs[-1][1].set_xlabel('Iterations', fontsize=AXIS_LABEL_FONTSIZE)

    for u in range(nb_use_cases):
        case = use_cases[u]
        axs[u][0].set_ylabel(case, fontsize=AXIS_LABEL_FONTSIZE)
        for name in bs_results[u].keys():
            color = colors_dict[name]
            label = name
            # BS coverage (statistical)
            ax = axs[u][0]
            y, perc_25, perc_75 = bs_results[u][name]
            x = np.arange(len(y))
            ax.plot(x, y, color=color, label=label)
            ax.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
            # Faulty BS coverage (statistical)
            ax = axs[u][1]
            y, perc_25, perc_75 = faulty_bs_results[u][name]
            x = np.arange(len(y))
            ax.plot(x, y, color=color, label=label)
            ax.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
        # once every methodology's results is displayed, adds the legend
        ax = axs[u][-1]
        legend = ax.legend(prop={'size': 10}, labelspacing=1.1, handletextpad=1.05, borderpad=1.05, borderaxespad=1.0)
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor('0.9')
        legend_frame.set_edgecolor('0.9')
    fig.tight_layout()
    return (fig, axs)


def plot_rq2_fs_results(
        use_cases: List[str],
        colors_dict: Dict[str, Tuple],
        knn_results: List[Dict[str, Union[Tuple, np.ndarray]]],
        fknn_results: List[Dict[str, Union[Tuple, np.ndarray]]],
        computation_steps: int = 200,
        x_index: int = 0
    ):
    '''Plots the knn results for RQ2.'''

    nb_use_cases = len(use_cases)

    fig, axs = plt.subplots(nrows=nb_use_cases, ncols=2, figsize=(10, 4*nb_use_cases), sharex=True, sharey='row')
    if nb_use_cases == 1:
        axs = [axs]
        [ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1) for ax in axs[0].flat]
    else:
        [ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1) for ax in axs.flat]

    axs[0][0].set_title(KNN_LABEL, fontsize=TITLE_LABEL_FONTSIZE)
    axs[0][1].set_title(FKNN_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
    axs[-1][0].set_xlabel('Iterations', fontsize=AXIS_LABEL_FONTSIZE)
    axs[-1][1].set_xlabel('Iterations', fontsize=AXIS_LABEL_FONTSIZE)

    x = np.arange((1 + x_index) * computation_steps - 1, 5000, computation_steps)

    for u in range(nb_use_cases):
        case = use_cases[u]
        axs[u][0].set_ylabel(case, fontsize=AXIS_LABEL_FONTSIZE)
        for name in knn_results[u].keys():
            color = colors_dict[name]
            label = name

            ax = axs[u][0]
            data = knn_results[u][name]
            if isinstance(data, np.ndarray):
                ax.plot(x, data[x_index:], color=color, label=label)
            else:
                y, perc_25, perc_75 = data
                ax.plot(x, y[x_index:], color=color, label=label)
                ax.fill_between(x, perc_25[x_index:], perc_75[x_index:], alpha=0.25, linewidth=0, color=color)
            legend = ax.legend(prop={'size': 10}, labelspacing=1.1, handletextpad=1.05, borderpad=1.05, borderaxespad=1.0)
            legend_frame = legend.get_frame()
            legend_frame.set_facecolor('0.9')
            legend_frame.set_edgecolor('0.9')
            ax = axs[u][1]
            data2 = fknn_results[u][name]
            if isinstance(data2, np.ndarray):
                ax.plot(x, data2[x_index:], color=color, label=label)
            else:
                y, perc_25, perc_75 = data2
                ax.plot(x, y[x_index:], color=color, label=label)
                ax.fill_between(x, perc_25[x_index:], perc_75[x_index:], alpha=0.25, linewidth=0, color=color)
            legend = ax.legend(prop={'size': 10}, labelspacing=1.1, handletextpad=1.05, borderpad=1.05, borderaxespad=1.0)
            legend_frame = legend.get_frame()
            legend_frame.set_facecolor('0.9')
            legend_frame.set_edgecolor('0.9')
    fig.tight_layout()
    for ax in axs[-1]:
        ax.set_xticks(np.arange((1 + x_index) * computation_steps, 5001, 1000))
    return (fig, axs)


def plot_rq3_results(data_lists: List[List[Dict]], colors_dict: Dict[str, Tuple[float]], xranges: List[Iterable], use_cases: List[str] = None, ylabels: List[str] = None):
    '''
    Convenient function that plots row-wise all the data provided, where the use-cases are displayed column-wise.
    The data is assumed to be lists of results per use-case.
    As such:
    - nrows is the length of @data_lists.
    - ncols is the maximum length of the lists in @data_lists.
    '''
    if use_cases is None:
        use_cases = np.arange(np.max([len(l) for l in data_lists]))

    nrows = len(data_lists)
    ncols = len(use_cases)

    sharex = 'all'
    if not np.all([len(r) == xranges[0] for r in xranges[1:]]):
        sharex = 'none'

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*nrows, 5*ncols), sharex=sharex, sharey='row')

    if ylabels is None:
        ylabels = ['' for _ in range(nrows)]
    for ax in axs.flat:
        ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1)

    for r, data in enumerate(data_lists):
        axs[r][0].set_ylabel(ylabels[r], fontsize=AXIS_LABEL_FONTSIZE)
        x = xranges[r]
        for c in range(ncols):
            ax = axs[r][c]
            case_dict = data[c]
            for name in case_dict.keys():
                to_plot = case_dict[name]
                color = colors_dict[name]
                label = name
                if isinstance(to_plot, np.ndarray):
                    assert len(x) == len(to_plot)
                    ax.plot(x, to_plot, color=color, label=label)
                else:
                    assert np.all([len(x) == len(tmp) for tmp in to_plot])
                    y, perc_25, perc_75 = to_plot
                    ax.plot(x, y, color=color, label=label)
                    ax.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
        ax = axs[r][-1]
        legend = ax.legend(prop={'size': 10}, labelspacing=1.1, handletextpad=1.05, borderpad=1.05, borderaxespad=1.0)
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor('0.9')
        legend_frame.set_edgecolor('0.9')
    for c in range(ncols):
        axs[0][c].set_title(use_cases[c], fontsize=TITLE_LABEL_FONTSIZE)
    fig.tight_layout()

    return (fig, axs)


#################################################################################################
################################## PERFORMANCE AND METRICS ######################################


def isin(array: np.ndarray, element: np.ndarray) -> Union[bool, np.ndarray]:
    return (array[:, None] == element).all(axis=-1).any(axis=1)


def isin_index(array: np.ndarray, element: np.ndarray):
    return next((i for i, j in enumerate((array[:, None] == element).all(axis=-1).any(axis=1)) if j), None)


##############################################################################
########################## RESULTS STORAGE/LOADING ###########################


def dump_dictionary(d: Dict[str, Union[np.ndarray, List]], filename: str):
    dict_to_dump = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            dict_to_dump[k] = v.tolist()
        else:
            dict_to_dump[k] = v
    with open(f'{filename.split(".")[0]}.json', 'w') as f:
        f.write(json.dumps(dict_to_dump))


def dump_dictionaries(dict_list: List[Dict[str, Union[np.ndarray, List]]], filenames: List[str]):
    assert len(filenames) == len(dict_list)

    for d, name in zip(dict_list, filenames):
        dump_dictionary(d, name)


def dump_results(dict_list: List[Dict[str, Iterable]], filenames: List[str]):
    assert len(filenames) == len(dict_list)

    for d, name in zip(dict_list, filenames):
        dict_to_dump = {}
        for k, v in d.items():
            # assert len(v) == 3, f'Three statistics are expected (found {len(v)}).'
            # assert np.all([len(t) == v[0] for t in v]), 'Three statistics of a result is malformed.'
            if isinstance(v, np.ndarray):
                dict_to_dump[k] = v.tolist()
            else:
                # must be a list of numpy arrays
                assert np.all([isinstance(t, np.ndarray) for t in v])
                dict_to_dump[k] = [t.tolist() for t in v]
        with open(f'{name.split(".")[0]}.json', 'w') as f:
            f.write(json.dumps(dict_to_dump))


def load_result(filepath: str):
    fp = f'{filepath}.json'
    assert os.path.exists(fp)
    try:
        with open(fp, 'r') as f:
            d = json.load(f)
    except:
        d = {}
    finally:
        for k in d.keys():
            v = d[k]
            assert isinstance(v, List)
            if isinstance(v[0], List):
                new_v = [np.array(l) for l in v]
            else:
                new_v = np.array(v)
            d[k] = new_v
        return d


def load_results(data_folder: str = 'data/', keys: List[str] = ['rq1', 'bs_cov', 'fbs_cov', 'knn_relative', 'fknn_relative']):
    '''
    Returns all the results per use-case-folder as a list of dictionaries ordered by use-case.
    Therefore, every list has a length of |use-cases|, where the elements are dictionaries of the results (whose keys are the names of the framework).
    '''
    if not data_folder.endswith('/'):
        data_folder += '/'
    use_cases = os.listdir(data_folder)
    all_results = [[] for _ in range(len(keys))]
    for k in range(len(keys)):
        for case in use_cases:
            all_results[k].append(load_result(f'{data_folder}{case}/{keys[k]}'))
    return all_results


##############################################################################
################################## MAIN ######################################


# exec(open('result_analysis.py').read())
if __name__ == '__main__':
    torch.set_num_threads(1)
    main_seed = 2021
    env_seed = 0


    ####################### Raw data loading #######################


    # LL
    ll_results = read_results_from_folder('results/ll/', include_final_states=True)
    ll_results.extend(read_results_from_folder('results/ll_mdpfuzz/', include_final_states=True))

    # TT
    ll_results.extend(read_results_from_folder('results/tt/', include_final_states=True))
    ll_results.extend(read_results_from_folder('results/ttns/', include_final_states=True))
    ll_results.extend(read_results_from_folder('results/tt_mdpfuzz/', include_final_states=True))

    BW_DEFAULT_USE_CASE = 'Bipedal Walker 0'
    bw_results = read_results_from_folder('results/bw/', include_final_states=True)
    bw_results.extend(read_results_from_folder('results/bw_mdpfuzz/', include_final_states=True))
    bw_results.extend(read_results_from_folder('results/bwrt/', include_final_states=True))

    first_results = ll_results + \
        [d for d in bw_results
            if d['config']['use_case'] == BW_DEFAULT_USE_CASE
        ]

    for d in first_results:
        if d['config']['use_case'] == BW_DEFAULT_USE_CASE:
            d['config']['use_case'] = 'Bipedal Walker'


    use_cases, method_names, colors_dict = color_data(first_results)
    print(method_names, use_cases)
    bw_cases, bw_names, bw_colors_dict = color_data(bw_results)
    print(bw_names, bw_cases)


    ####################### Analysis computation #######################


    rq1_data = compute_rq1_results(first_results)
    bs_cov, fbs_cov = compute_rq2_bs_results(first_results)
    # stores the results of the analysis
    folder = 'data'
    for case in use_cases:
        sub_folder = f'{folder}/{case}'
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)

    dump_results(rq1_data, [f'{folder}/{case}/rq1' for case in use_cases])
    dump_results(bs_cov, [f'{folder}/{case}/bs_cov' for case in use_cases])
    dump_results(fbs_cov, [f'{folder}/{case}/fbs_cov' for case in use_cases])

    # knn computation of the final states
    computation_steps, k = 50, 3
    use_cases, knn_results, fknn_results, _xstarts, colors_dict = compute_mean_knn(first_results, knn=k, computation_steps=computation_steps)
    relative_knns = [compute_relative_performance(results_dict) for results_dict in knn_results]
    relative_fknns = [compute_relative_performance(results_dict) for results_dict in fknn_results]
    dump_results(knn_results, [f'{folder}/{case}/knn' for case in use_cases])
    dump_results(fknn_results, [f'{folder}/{case}/fknn' for case in use_cases])
    dump_results(relative_knns, [f'{folder}/{case}/knn_relative' for case in use_cases])
    dump_results(relative_fknns, [f'{folder}/{case}/fknn_relative' for case in use_cases])

    # RQ3: only BW data with all behavior spaces
    rq3_data = compute_rq1_results(bw_results)
    rq3_bs_cov, rq3_fbs_cov = compute_rq2_bs_results(bw_results)

    bw_folder = 'data/rq3'
    if not os.path.exists(bw_folder):
        os.mkdir(bw_folder)
    for case in bw_cases:
        sub_folder = f'{bw_folder}/{case}'
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)

    dump_results(rq3_data, [f'{bw_folder}/{case}/rq1' for case in bw_cases])
    dump_results(rq3_bs_cov, [f'{bw_folder}/{case}/bs_cov' for case in bw_cases])
    dump_results(rq3_fbs_cov, [f'{bw_folder}/{case}/fbs_cov' for case in bw_cases])
    # knn computation of the final states
    bw_cases, rq3_knn_results, rq3_fknn_results, _xstarts, bw_colors_dict = compute_mean_knn(bw_results, knn=k, computation_steps=computation_steps)
    rq3_relative_knns = [compute_relative_performance(results_dict) for results_dict in rq3_knn_results]
    rq3_relative_fknns = [compute_relative_performance(results_dict) for results_dict in rq3_fknn_results]
    dump_results(rq3_knn_results, [f'{bw_folder}/{case}/knn' for case in bw_cases])
    dump_results(rq3_fknn_results, [f'{bw_folder}/{case}/fknn' for case in bw_cases])
    dump_results(rq3_relative_knns, [f'{bw_folder}/{case}/knn_relative' for case in bw_cases])
    dump_results(rq3_relative_fknns, [f'{bw_folder}/{case}/fknn_relative' for case in bw_cases])


    ####################### Plotting #######################

    fig1, axs1 = plot_rq1_results(use_cases, colors_dict, rq1_data)
    for ax in axs1.flat:
        legend = ax.legend_
        if legend is not None:
            for line in legend.get_lines():
                plt.setp(line, linewidth=4)
    fig1.savefig('test_rq1.png')

    fig2, axs2 = plot_rq2_bs_results(use_cases, colors_dict, bs_cov, fbs_cov)
    axs2[0][-1].legend_ = None
    axs2[-1][-1].legend_ = None
    legend = axs2[1][-1].legend_
    for line in legend.get_lines():
        plt.setp(line, linewidth=4)
    fig2.savefig('test_rq22_bs.png')

    # this final state diversity is not shown for the initialization data (i.e., first 1000 iterations) since the values vary a lot.
    x_index = 19 # it means that the plotting starts at the (1 + 19) * 50 = 1000th iteration
    fig3, axs3 = plot_rq2_fs_results(use_cases, colors_dict, relative_knns, relative_fknns, computation_steps, x_index)
    for ax in axs3[0]:
        ax.legend_ = None
    for ax in axs3[-1]:
        ax.legend_ = None
    axs3[1][-1].legend_ = None
    fig3.savefig('test_rq22_fs.png')


    # the last plotting function does not support such plotting starting index: we thus have to shorten the data before
    x = np.arange((1 + x_index) * computation_steps - 1, 5000, computation_steps)
    for dd in rq3_relative_knns:
        for k, v in dd.items():
            dd[k] = v[x_index:]
    for dd in rq3_relative_fknns:
        for k, v in dd.items():
            dd[k] = v[x_index:]

    fig4, axs4 = plot_rq3_results(
        [rq3_data, rq3_bs_cov, rq3_fbs_cov, rq3_relative_knns, rq3_relative_fknns],
        colors_dict,
        [np.arange(5000), np.arange(5000), np.arange(5000), x, x],
        ['$Distance$ and $Hull$ $angle$', '$Torque$ and $Jump$', '$Hip$ $angles$', '$Hip$ $speeds$'],
        ['#Faults', '#Behaviours', '#Faulty Behaviours', 'FS Diversity', 'FFS Diversity']
    )
    fig4.set_figwidth(15)
    fig4.set_figheight(15)
    fig4.tight_layout()
    # removes redundant x labels
    for i in [0, 1, 3]:
        for ax in axs4[i]:
            ax.set_xticklabels([])
    # removes redundant legends
    for i in [0, 2, 3, 4]:
        axs4[i][-1].legend_ = None
    fig4.savefig('test_rq3.png')