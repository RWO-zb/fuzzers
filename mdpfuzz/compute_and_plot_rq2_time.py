import os

import numpy as np
from typing import Dict, List

from common import METHOD_LABELS, USE_CASES, USE_CASE_LABELS_DICT
from data_management import get_data_from_experiment, load_results, store_dict
from data_processing import compute_time_results
from plotting import create_bar_plots, get_colors, get_method_colors

'''
This file processes the data from data_rq2/ and performs time analysis of the paper.
It should return a .png similar to Figure 3.
'''

# computes and stores results or loads them
LOAD = False

if __name__ == '__main__':
    colors = get_method_colors()
    # variables to find data
    method_names = ['fuzzer', 'mdpfuzz', 'rt']
    use_case_keys = ['acas', 'bw', 'carla', 'cart', 'coop', 'll', 'tt']

    #################### 1st metric: Fault Discovery ###################
    rq2_data_folder = 'data_rq2'
    rq2_results_folder = 'results_rq2'

    if not os.path.isdir(rq2_results_folder):
        os.mkdir(rq2_results_folder)

    method_results_list: List[Dict] = []
    labels = []
    for name in method_names:
        results_path = f'{rq2_results_folder}/time_{name}'
        if not LOAD:
            dict_list = []
            for case in use_case_keys:
                try:
                    dict_list.append(
                        get_data_from_experiment(
                            f'{rq2_data_folder}/{case}/',
                            config_pattern=f'{name}.*_config.json')
                    )
                    if case not in labels:
                        labels.append(case)
                except:
                    print("No data for {} {}.".format(name, case))
            results = compute_time_results(dict_list)
            store_dict(results_path, results)
        else:
            results = load_results(results_path)
            use_cases_found = list(results.keys())
            for u in use_cases_found:
                if u not in labels:
                    labels.append(u)
            print("Loaded data for {} and {}.".format(name, len(use_cases_found)))
        method_results_list.append(results)

    print("Use cases to plot:", labels)
    plot_data = {}
    print('checking time results\' internal structure...')
    # re-organizes the results per use-case
    # checks results' internal structure
    keys_list = [list(d.keys()) for d in method_results_list]
    if not LOAD:
        labels = [USE_CASE_LABELS_DICT[k] for k in labels]
    [l.sort() for l in keys_list]
    assert np.all([l == labels for l in keys_list]), [l for l in keys_list]

    time_keys = ['run', 'test', 'cov']
    for d in method_results_list:
        # time results of all the use-cases as dictionaries
        time_results: List[Dict] = list(d.values())
        # checks that all those dicitonaries have the correct keys
        assert np.all([list(tmp.keys()) == time_keys for tmp in time_results])
    print('Result checking done.')
    for r in labels:
        data = []
        numerical_data = []
        for i in range(len(method_results_list)):
            tmp = []
            numerical_tmp = []
            times = method_results_list[i][r]
            for (m, e) in times.values():
                m = m / 60
                e = e / (2 * 60)
                numerical_tmp.append((m, 2 * e))
                if m != 0:
                    tmp.append((f'{m:.2f}', f'{e:.1f}'))
            data.append(tuple(tmp))
            numerical_data.append(tuple(numerical_tmp))
        plot_data[r] = numerical_data
    time_colors = { k: v for k, v in zip(['Total', '$\\pi$-Env.', 'Cov.'],  get_colors(10, 'tab10')[:3]) }
    fig, axs = create_bar_plots(plot_data, METHOD_LABELS, time_colors)
    fig.tight_layout()
    fig.savefig('rq2_time.png')
    print("Time analysis done (see \"rq2_time.png\").")

