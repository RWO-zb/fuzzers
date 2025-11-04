import time
import sys
import os
import json
import numpy as np
import pandas as pd

from typing import Tuple, Union, Dict, List


EXPERIMENT_SEEDS = [2021, 42, 2023, 20, 0, 10, 4, 2006, 512, 1453]
POP_SIZES = [100, 250, 500]
ITERATIONS = [50, 20, 10]


###############################################################################################
################################## CELL AND GRID HELPERS ######################################

def compute_cell(behavior: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    cell = []
    for b, v in zip([xedges, yedges], behavior):
        if v < b[1]:
            cell.append(0)
        elif v >= b[-2]:
            cell.append(len(b) - 1)
        else:
            cell.append(np.argmax(v < b) - 1)
    return np.array(cell)


def compute_cells(behaviors: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    cells = []
    for behavior in behaviors:
        cell = []
        for b, v in zip([xedges, yedges], behavior):
            if v < b[1]:
                cell.append(0)
            elif v >= b[-2]:
                cell.append(len(b) - 1)
            else:
                cell.append(np.argmax(v < b) - 1)
        cells.append(np.array(cell))
    return np.array(cells)


def compute_grid_edges(bins: int = 50, mins: np.ndarray = None, maxs: np.ndarray = None, behaviors: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Returns the cell edges structure used by the regular grid.
    The edges are bins + 1 numpy arrays where the first and last value are the minimum and maximum, respectively.
    Besides, the extrema can be provided instead of the behaviors.
    '''
    # I don't know how to handle this... the function's signature is soo bad
    if behaviors is None:
        assert (mins is not None) and (maxs is not None), 'Either behaviors or extrema have to be provided.'

    if mins is None:
        mins = np.min(behaviors, axis=0)
    if maxs is None:
        maxs = np.max(behaviors, axis=0)

    edges = np.array([np.linspace(min, max, num=(bins + 1)) for min, max in zip(mins, maxs)])
    return edges, mins, maxs


def get_histogram(behaviors: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> np.ndarray:
    '''Returns the histogram of the behaviors (i.e., the behavior points distribution in the space).'''
    # the issue is that some behaviors found during the search might be outside of the edges.
    # this is handled by the archive, but not here.
    return np.histogram2d(behaviors[:, 0], behaviors[:, 1], bins=(xedges, yedges))[0]


#################################################################################################
################################## RESULTS READING HELPERS ######################################


def process_txt_log(filename: str):
    '''Reads a log file (.txt, lines of "key:value,") of an execution and returns the results as a DataFrame (a row describes the log of an iteration).'''
    if not filename.endswith('.txt'):
        filename += '.txt'
    assert os.path.isfile(filename)

    t0 = time.time()
    dicts = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            # dirty...
            try:
                splits = line.split(',')
                str_dict = dict(s.strip().split(':') for s in splits)
                dicts.append({k: float(v) for k, v in str_dict.items()})
            except:
                print(f'ERROR_TXT_LOG_PROCESSING for "{line}".', file=sys.stderr)

    df = pd.DataFrame.from_records(dicts)
    if 'oracle' in df.columns:
        df['oracle'] = df['oracle'].astype(bool)
    process_time = time.time() - t0
    return df, process_time


def retrieve_result(filepath: str, **kwargs) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    '''
    Returns the results of a testing methodology at @filepath as a dictionary of:
    - 3 numpy arrays (inputs, behaviors and cells).
    - DataFrame of the logs.
    - DataFrame of the internal data of the Framework class.
    - Dictionary of the (experimental) configuration.

    Raise an error if any of the expected file is missing.
    Kwargs:
    - "include_final_states", the dictionary has the latter at the key 'final_states'.
    - "is_ns"; the dictionaries have particular NS logs at the key 'ns_logs'.
    '''
    filepaths = [f'{filepath}_{k}.txt' for k in ['inputs', 'behaviors', 'cells']]
    filepaths.append(f'{filepath}_logs.txt')
    filepaths.append(f'{filepath}_data.csv')

    if not np.all([os.path.exists(fp) for fp in filepaths]):
        raise FileNotFoundError('One of the required result file is missing.')

    result = {k: np.loadtxt(f'{filepath}_{k}.txt', delimiter=',') for k in ['inputs', 'behaviors', 'cells']}
    result['logs'] = process_txt_log(f'{filepath}_logs.txt')[0]
    result['data'] = pd.read_csv(f'{filepath}_data.csv')
    try:
        with open(f'{filepath}_config.json', 'r') as f:
            result['config'] = json.load(f)
    except:
        result['config'] = {}
        # print(f'No configuration found at {filepath}.')

    include_final_states = kwargs.get('include_final_states', False)
    if include_final_states:
        final_states_fp = f'{filepath}_final_states.txt'
        if os.path.exists(final_states_fp):
            result['final_states'] = np.loadtxt(final_states_fp, delimiter=',')
    is_ns = kwargs.get('is_ns', False)
    if is_ns:
        ns_logs_fp = f'{filepath}_ns_logs.txt'
        if os.path.exists(ns_logs_fp):
            result['ns_logs'] = process_txt_log(ns_logs_fp)[0]
    return result


def read_results_from_folder(results_folder: str, **kwargs) -> List[Dict]:
    '''
    Returns all the results found in @results_folder as a list of dictionaries.
    Kwargs:
        - "include_final_states"; the dictionaries have the latter at the key 'final_states'.
        - "is_ns"; the dictionaries have particular NS logs at the key 'ns_logs'.
    '''
    assert os.path.isdir(results_folder)
    if not results_folder.endswith('/'):
        results_folder += '/'

    results_filepathes = [results_folder + fp for fp in set(f.split('_')[0] for f in os.listdir(results_folder))]

    config = kwargs.get('config', {})
    name_key = kwargs.get('name_key', None)

    dicts = []

    for fp in results_filepathes:
        config_fp = fp + '_config.json'
        if os.path.exists(config_fp):
            with open(config_fp, 'r') as f:
                f_config: dict = json.load(f)

            if not all(f_config.get(k) == config[k] for k in config.keys()):
                continue
            if (name_key is not None) and (not name_key in f_config['name']):
                continue
        try:
            d = retrieve_result(fp, **kwargs)
            dicts.append(d)
        except FileNotFoundError:
            pass
    return dicts