import os

import numpy as np
from typing import Dict, List

from common import METHOD_LABELS, USE_CASE_LABELS_DICT, USE_CASES
from data_management import get_logs, store_results, load_results
from data_processing import compute_statistical_fault_discovery_results
from plotting import get_gamma_colors, get_method_colors, plot_k_g_analysis

'''
This file retrieves the data from the parameter sensibility analysis \
    of MDPFuzz and plots Figure 6.
'''

# computes and stores results or loads them
LOAD = False

if __name__ == '__main__':
    colors = get_method_colors()
    # variables to find data
    method_names = ['fuzzer', 'mdpfuzz', 'rt']
    use_case_keys = ['acas', 'bw', 'carla', 'cart', 'coop', 'll', 'tt']


    rq3_data_folder = 'data_rq3'
    rq3_results_folder = 'results_rq3'

    K = [6, 8, 10, 12, 14]
    G = [0.05, 0.1, 0.15, 0.2]
    TAU = 0.01
    if not os.path.isdir(rq3_results_folder):
        os.mkdir(rq3_results_folder)

    rq3_results = []
    labels = []
    for k in K:
        print("Processing results for K = {} ...".format(k))
        for g in G:
            print("... With gamma = {}".format(g))
            results_path = f'{rq3_results_folder}/mdpfuzz_{k}_{TAU}_{g}'
            d = {
                'k': k,
                'tau': TAU,
                'gamma': g,
                'name': 'mdpfuzz'
            }
            if not LOAD:
                # list of logs per use-case
                logs = []
                use_case_found = []
                for key in use_case_keys:
                    prefix = f'{key}_{k}_{TAU}_{g}_'
                    config_logs = get_logs(rq3_data_folder, prefix)
                    # if key == 'carla':
                    #     assert len(config_logs) == 3, len(config_logs)
                    # else:
                    #     assert len(config_logs) == 5, f'{prefix}: {len(config_logs)}'
                    if config_logs == []:
                        print("No data for config {} {} {}".format(key, k, g))
                    else:
                        logs.append(config_logs)
                        use_case_found.append(key)
                        if key not in labels:
                            labels.append(key)
                    # logs.append(config_logs)
                # assert len(logs) == 7, f'{k}_{TAU}_{g}_'
                fault_results = compute_statistical_fault_discovery_results(logs)
                results = store_results(use_case_found, fault_results, results_path, d)
            else:
                results = load_results(results_path)
                use_cases_found = list(results.keys())
                for u in use_cases_found:
                    if (u in use_case_keys) and (u not in labels):
                        labels.append(u)
            rq3_results.append(results)
        print("Processing K = {} done!".format(k))
    gamma_dict = {g: c for g, c in zip(G, get_gamma_colors())}
    fig, axs = plot_k_g_analysis(
        labels,
        rq3_results,
        gamma_dict,
        y_axis='k',
        use_case_labels=[USE_CASE_LABELS_DICT[k] for k in labels],
        filename='rq3.png'
    )
    print("Parameter analysis done (see \"rq3.png\").")
