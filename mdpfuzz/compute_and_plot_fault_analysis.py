import json
import os
import sys

from matplotlib.gridspec import GridSpec
from data_management import get_logs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
from sklearn.manifold import TSNE

sys.path.append('methods/src')
from logger import Logger


FOLDER_TO_LABEL = {
    'fuzzer': 'Fuzzer-R',
    'mdpfuzz': 'MDPFuzz-R',
    'rt': 'RT'
}

FOLDER_TO_COLOR = {
    'fuzzer': 'fire engine red',
    'mdpfuzz': 'deep sky blue',
    'rt': 'shamrock green'
}

FOLDER_TO_TITLE = {
    f: t for
        f, t in zip(
            ['acas', 'bw', 'carla', 'cart', 'coop', 'll', 'tt'],
            ['ACAS Xu', 'Bipedal Walker', 'CARLA', 'Cart Pole', 'Coop Navi', 'Lunar Lander', 'Taxi']
            )
}


def store_dict(filepath: str, result_dict: Dict[str, np.ndarray]):
    dict_to_store = {}
    for k, v in result_dict.items():
        dict_to_store[k] = v.tolist()
    filename = filepath.split('.json')[0]
    with open(f'{filename}.json', 'w') as f:
        f.write(json.dumps(dict_to_store))


def load_dict(filepath: str):
    d = {}
    with open(filepath, 'r') as f:
        d = json.load(f) # type: dict
    if d == {}:
        return

    for k in d.keys():
        v = d[k]
        if isinstance(v, List):
            assert np.all(isinstance(l, List) for l in v)
            new_v = np.array([np.array(l) for l in v])
        else:
            new_v = v
        d[k] = new_v
    return d


def load_log_file(filename: str):
    if not filename.endswith('.txt'):
        filename += '.txt'
    if not os.path.isfile(filename):
        raise FileNotFoundError("\"{}\" not found.".format(filename))
    logger = Logger(filename)
    df = logger.load_logs()
    return df


def get_logs(folder: str, prefix: str = ''):
    if not folder.endswith('/'):
        folder += '/'
    return [load_log_file(folder + f) for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('_logs.txt')]


def get_faults(df_list: List[pd.DataFrame], include_unique_faults_only: bool = True):
    data = []
    for i, df in enumerate(df_list):
        # print('--------------- DEBUG DF {:02d} ---------------'.format(i))
        inputs = np.vstack([arr for arr in df['input']])
        unique_input_indices = np.unique(inputs, axis=0, return_index=True)[1]
        # print('Unique: {:04d} / {:04d}'.format(len(unique_input_indices), len(inputs)))
        oracles = df['oracle'].to_numpy()
        # print('Faults: {:04d}'.format(sum(oracles)))
        if include_unique_faults_only:
            m1, m2 = np.zeros(len(inputs)), np.zeros(len(inputs))
            m1[unique_input_indices] = True
            m2[oracles] = True
            mask = np.stack([m1, m2], axis=1).all(axis=1)
            data.append(inputs[mask])
        else:
            data.append(inputs[oracles])
    return data


def plot(data: Dict[str, np.ndarray], figsize=(8,6)):
    fig, ax = plt.subplots(figsize=figsize)

    for name, faults in data.items():
        ax.scatter(
            faults[:, 0],
            faults[:, 1],
            color='xkcd:'+FOLDER_TO_COLOR[name],
            label=FOLDER_TO_LABEL[name],
            s=5)

    legend = ax.legend(
        prop={'size': 12},
        labelspacing=1.0,
        handletextpad=1.0,
        borderpad=0.55,
        borderaxespad=0.7)
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('0.9')
    legend_frame.set_edgecolor('0.9')
    for handle in legend.legendHandles:
        handle.set_sizes([50.0])
    return fig, ax



def plots(data_list: List[Dict[str, np.ndarray]], point_size: int = 5):
    num_plots = len(data_list)
    # size = 6
    # fig, axs = plt.subplots(nrows=num_plots, ncols=1, figsize=(size, size*num_plots))
    fig = plt.figure(figsize=(30, 10))
    gs = GridSpec(2, (2 * num_plots), figure=fig)

    index = int(num_plots / 2)
    axs = []
    size = 2
    for i in range(0, index + 2 * size + 1, size):
        axs.append(fig.add_subplot(gs[0, i:(i+size)]))
    for i in range(1, index + 2 * size, size):
        axs.append(fig.add_subplot(gs[1, i:(i+size)]))



    for i in range(num_plots):
        ax = axs[i]
        data = data_list[i]
        for name, faults in data.items():
            ax.scatter(
                faults[:, 0],
                faults[:, 1],
                color='xkcd:'+FOLDER_TO_COLOR[name],
                label=FOLDER_TO_LABEL[name],
                s=point_size)

    # adds legend to one plot
    ax = axs[2]
    legend = ax.legend(
        prop={'size': 13},
        labelspacing=1.0,
        handletextpad=1.0,
        borderpad=0.55,
        borderaxespad=0.7)
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('0.9')
    legend_frame.set_edgecolor('0.9')
    for handle in legend.legendHandles:
        handle.set_sizes([50.0])
    return fig, axs


if __name__ == '__main__':
    method_names = ['fuzzer', 'mdpfuzz', 'rt']
    use_case_keys = ['acas', 'bw', 'carla', 'cart', 'coop', 'll', 'tt']
    rq2_data_folder = 'data_rq2'

    dict_list = []

    labels = []
    for use_case_folder in use_case_keys:
        print(f"Processing {use_case_folder} ...")
        try:
            indices = [0]
            results = []
            for name in method_names:
                method_logs = get_logs(f'{rq2_data_folder}/{use_case_folder}/', name)


                fault_list = get_faults(method_logs, False)
                fault_array = np.vstack(fault_list)
                # print(fault_array.shape)

                indices.append(indices[-1]+len(fault_array))
                results.append(fault_array)

            indices[-1] -= 1
            # print(f"Indice slices: {indices}")
            acc_results = np.vstack(results)
            print(f"{len(acc_results)} data points collected.")
            projections = TSNE(
                n_components=2,
                init='random',
                random_state=0
                ).fit_transform(acc_results)
            data = {}
            for i, mn in enumerate(method_names):
                data[mn] = projections[indices[i]:indices[i+1]]
            # saves the data
            store_dict(
                f"faults_distribution_{use_case_folder}.json",
                data)
            dict_list.append(data)
            labels.append(use_case_folder)
        except Exception as e:
            # data = load_dict(
            # f"faults_distribution_{use_case_folder}.json"
            # )
            # dict_list.append(data)
            # labels.append(use_case_folder)
            print("Something went wrong with use case {}:".format(use_case_folder))
            print('"{}".'.format(e))

    fig, axs = plots(dict_list)
    for ax, folder in zip(axs, labels):
        ax.set_title(FOLDER_TO_TITLE[folder], fontsize=22)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    fig.tight_layout()
    f = f"fault_distribution.png"
    fig.savefig(f)
    print(f"Fault distribution analysis done (see \"{f}\").")