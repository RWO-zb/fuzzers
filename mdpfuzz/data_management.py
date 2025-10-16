import sys
import os
import json
import re
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Union


sys.path.append('methods/src')
from logger import Logger


###### DATA READING #####

def load_log_file(filename: str):
    '''Returns a log file as a pd.DataFrame.'''
    if not filename.endswith('.txt'):
        filename += '.txt'
    if not os.path.isfile(filename):
        raise FileNotFoundError("\"{}\" not found.".format(filename))

    logger = Logger(filename)
    df = logger.load_logs()

    # The very first runs of Fuzzer with CARLA did not record the inputs correctly (in the logs).
    # Given the very (very) low probabilitiy of redundant inputs for that use case (subspace of R^406...),
    # we decided to not repeat the experiments but rather bypass this "technical issue".
    if 'fuzzer' in filename and 'carla' in filename:
        # print(f'changing inputs found in file {filename}')
        df['input'] = np.arange(len(df))
    return df




def get_logs(folder: str, prefix: str = ''):
    '''Return the results of log files starting with @prefix (found in @folder) as a list of pd.DataFrames.'''
    if not folder.endswith('/'):
        folder += '/'
    return [load_log_file(folder + f) for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('_logs.txt')]


def _read_execution_parameters(filepath: str) -> Dict[str, Union[str, float]]:
    filepath = filepath.split('.json')[0] + '.json'
    with open(filepath, 'r') as f:
        config: Dict = json.load(f)
    param_dict = {
        'k': config['k'],
        'gamma': config['gamma'],
        'tau': config['tau'],
        'use_case': config['use_case'],
        'name': config['name']
    }
    return param_dict


def get_data_from_experiment(folder: str, config_pattern='^\d+_config\.json$', log_suffix: str ='_logs.txt') -> Dict[str, Union[str, float, List[pd.DataFrame]]]:
    '''
        Reads the results of an experiment from a specified folder as a dictionary that contains the parameters of the experiments
        and the list of Pandas DataFrames parsed from the log files.

    Parameters:
        - folder (str): The path to the folder containing the data of the experiment.
        - config_pattern (Union[str, re.Pattern], optional): Pattern to match the filepathes found (default to '_config.json').
        - log_suffix (str, optional): Suffix of the log files (default to '_logs.txt').

    Returns:
        Dict[str, Union[str, float, List[pd.DataFrame]]]: A dictionary summarizing the experiment's parameters and results.
        The minimum set of keys is "name", "K", "tau", "gamma", "results" and "use_case".
        The test budget is indicated as the value whose key is either "test_budget" or "test_budget_in_seconds".
        Additional keys can for example be "init_budget".

    Raises:
        AssertionError: If the provided 'folder' is not a valid directory.

    Example:
        ```
        folder_path = '/path/to/log/files/'
        log_data = get_data_from_experiment(folder_path, config_pattern='.*_config.json')
        ```
    '''
    assert os.path.isdir(folder), folder
    if not folder.endswith('/'):
        folder += '/'

    pattern_obj = re.compile(config_pattern)
    filepathes = [folder + f for f in os.listdir(folder) if pattern_obj.match(f)]
    if len(filepathes) == 0:
        raise Exception(f'No configuration file found in folder {folder}')
    config = _read_execution_parameters(filepathes[0])

    for fp in filepathes[1:]:
        tmp = _read_execution_parameters(fp)
        assert np.all([config[k] == tmp[k] for k in tmp.keys()]), 'Parameters are not all the same.'

    log_filepathes = [f.replace('_config.json', log_suffix) for f in filepathes]
    config['data'] = [load_log_file(f) for f in log_filepathes]
    return config


##### DATA STORAGE #####


def store_dict(filepath: str, result_dict: Dict[str, Union[str, float, int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]):
    '''Stores statistical results at the given @filename.'''
    dict_to_store = {}
    for k, v in result_dict.items():
        if isinstance(v, tuple):
            assert np.all([isinstance(t, np.ndarray) for t in v])
            dict_to_store[k] = [t.tolist() for t in v]
        else:
            dict_to_store[k] = v
    filename = filepath.split('.json')[0]
    with open(f'{filename}.json', 'w') as f:
        f.write(json.dumps(dict_to_store))


def load_dict(filepath: str) -> Union[str, float, int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Loads stored results.
    It return an empty dictionnary if the dumped dictionnary is malformed.
    '''
    filepath = filepath.split('.json')[0] + '.json'
    if not os.path.isfile(filepath):
        print("\"{}\" not found.".format(filepath))
        print("Empty dictionary returned.")
        return {}
    try:
        with open(filepath, 'r') as f:
            d = json.load(f)
    except:
        d = {}
    finally:
        for k in d.keys():
            v = d[k]
            if isinstance(v, List):
                assert np.all(isinstance(l, List) for l in v)
                new_v = [np.array(l) for l in v]
            else:
                new_v = v
            d[k] = new_v
        return d


def store_results(
        use_cases: List[str],
        results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        output_path: str,
        results_dict: Dict = {}
    ):
    '''
    Stores the results of a testing method.
    The results are assumed to be sorted w.r.t @use_cases!
    It stores a dictionnary whose keys are the use-cases.
    The latter can be provided.
    '''
    tmp_dict = results_dict.copy()
    for k, t in zip(use_cases, results):
        tmp_dict[k] = t
    store_dict(output_path, tmp_dict)
    return tmp_dict


def load_results(path: str) -> Dict[str, Union[str, float, int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    '''Load stored results of a testing method as a dictionnary of statistical results indexed by the use-cases.'''
    return load_dict(path)


def find_files(directory, extension):
    file_paths = []

    # Walk through the directory tree
    for root, directories, files in os.walk(directory):
        for file in files:
            # Check if the file ends with the specified extension
            if file.endswith(extension):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Append the file path to the list
                file_paths.append(file_path)

    return file_paths


def aggregate_iterations(df_list: List[pd.DataFrame], time_step: float = 1.0):
    '''
    Retuns the accumulation of the number of iterations per @time_step.
    '''
    results = []
    for df in df_list:
        times = df['time'].to_numpy()

        time = time_step
        iteration_accumulator = []
        num_iterations = 0

        for t in times:
            if t <= time:
                num_iterations += 1
            elif t > time:
                iteration_accumulator.append(num_iterations)
                time += time_step
                num_iterations += 1

        iteration_accumulator.append(num_iterations)
        results.append(np.array(iteration_accumulator))
    return results


def process_dataframes(df_list: List[pd.DataFrame]):
    '''
    Processes a list of pd.DataFrames.
    The processing consists in adding relative timestamps (column 'time'), which thus removes the first iteration.
    '''
    results = []
    for df in df_list:
        t0 = df['run_time'][0]
        df['time'] = df['run_time'] - t0
        results.append(df.tail(len(df) - 1))
    return results


def aggregate_timestamped_dataframes(df_list: List[pd.DataFrame]):
    concatenated_df = pd.concat(df_list)
    sorted_df = concatenated_df.sort_values(by='time')
    print('sorted length', len(sorted_df))
    sorted_df.reset_index(drop=True, inplace=True)
    print('after index reseting', len(sorted_df))
    return sorted_df