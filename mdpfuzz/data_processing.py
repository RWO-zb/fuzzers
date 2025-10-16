import time
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict


def compute_statistics(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Helper that computes statistical results from a set of results.'''
    if not isinstance(data, np.ndarray):
        data: np.ndarray = np.array(data)
    y = np.median(data, axis=0)
    perc_25 = np.percentile(data, 25, axis=0)
    perc_75 = np.percentile(data, 75, axis=0)
    return y, perc_25, perc_75


##### 1st Metric / Analysis: Fault Discovery #####


def compute_faults_distinct_inputs(oracles: List[np.ndarray]):
    '''
    Optimized version to compute fault discovery results where no additional checking is performed.
    Precisely, the function counts faults and return the list of the accumulation.
    '''
    results = []
    for arr in oracles:
        fault_accumulator = []
        num_faults = 0
        for o in arr:
            num_faults += int(o)
            fault_accumulator.append(num_faults)
        results.append(np.array(fault_accumulator))
    return results


def count_faults_stochastic_executions(inputs_list: List[np.ndarray], oracles_list: List[np.ndarray]):
    '''
    Counts the number of faults for a list of inputs and oracle flags where executions are not assumed to be deterministic.
    It addresses the CARLA use-case.
    As such, redundant inputs are not discarded.
    Instead, the function stores the fault-revealing inputs and counts a fault if the current input has not been already recorded.
    '''
    results = []
    for inputs, oracles in zip(inputs_list, oracles_list):
        assert len(inputs) == len(oracles)
        fault_revealing_inputs = []
        fault_accumulator = []
        num_faults = 0
        for i in range(len(oracles)):
            o = oracles[i]
            if o:
                tmp = inputs[i].tolist()
                try:
                    index = fault_revealing_inputs.index(tmp)
                    # print(f'Already counted! (index: {index})')
                except:
                    num_faults += 1
                    fault_revealing_inputs.append(tmp)
            fault_accumulator.append(num_faults)
        assert num_faults == len(fault_revealing_inputs), 'stochastic executions?!'
        # print(f'Found {num_faults} faults in total and detects {len(fault_revealing_inputs)} fault-revealing inputs.')
        results.append(np.array(fault_accumulator))
    return results




def compute_fault_discovery_results(df_list: List[pd.DataFrame]) -> Tuple[np.ndarray]:
    '''
    Processes experimental data for a single use-case (as a list of pd.DataFrames) and computes the results.
    Precisely, the processing consists of counting non-redundant faults.
    To ease plotting, the results are extended to the longest execution.
    '''
    inputs, oracles = [], []
    for df in df_list:
        inputs.append(np.vstack(df['input']))
        oracles.append(df['oracle'].to_numpy())
    t0 = time.time()
    if not np.all([len(arr) == len(np.unique(arr, axis=0)) for arr in inputs]):
        # print('WARNING: redundant inputs found.')
        tmp = count_faults_stochastic_executions(inputs, oracles)
    else:
        tmp = compute_faults_distinct_inputs(oracles)
    # print(f'process time: {(time.time() - t0):.2f}s.')
    if tmp == []:
        print("Warning: not fault found in the list of pd.DataFrames.")
    max_length = np.max([len(l) for l in tmp]) if tmp != [] else 0
    for i in range(len(tmp)):
        arr = tmp[i]
        last_value = arr[-1]
        arr_length = len(arr)
        if arr_length < max_length:
            tmp[i] = np.array([v for v in arr] + [last_value for _ in range(max_length - len(arr))])
    return compute_statistics(tmp)


def compute_statistical_fault_discovery_results(data: List[List[pd.DataFrame]]) -> List[Tuple[np.ndarray]]:
    '''Aggregates statistical results for fault discovery by calling compute_fault_discovery_results(.) for each DataFrame list.'''
    results = []
    for df_list in data:
        results.append(compute_fault_discovery_results(df_list))
    return results


##### 2nd Metric: Time Analysis #####


def process_time_data(data: pd.DataFrame):
    '''Returns the total times for a pd.DataFrame.'''
    run_times = data['run_time'].to_numpy()
    total_run_time = run_times[-1] - run_times[0]
    total_test_exec_time = np.sum(data['test_exec_time'].to_numpy())
    coverage_times = data['coverage_time']
    if not pd.isnull(coverage_times).all():
        total_coverage_time = np.nansum(coverage_times.to_numpy())
    else:
        total_coverage_time = 0
    return total_run_time, total_test_exec_time, total_coverage_time


def compute_time_results(data: List[Dict]) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    '''Returns a list of dictionaries for a given method.'''
    use_cases: List[str] = np.unique([d['use_case'] for d in data]).tolist()
    results = {}

    for case in use_cases:
        method_data: List[pd.DataFrame] = [d['data'] for d in data if d['use_case'] == case]
        tmp = {}
        if len(method_data) == 0:
            print(f'No result found for use-case {case}')
            continue

        assert len(method_data) == 1, 'More than one match for experiment {}'.format(case)
        # processes the data
        tot_run_times, tot_test_times, tot_cov_times = [], [], []
        for df in method_data[0]:
            tot_run_time, tot_test_time, tot_cov_time = process_time_data(df)
            tot_run_times.append(tot_run_time)
            tot_test_times.append(tot_test_time)
            tot_cov_times.append(tot_cov_time)
        # computes statistics and reorganizes the results for convenient plotting
        for key, times in zip(['run', 'test', 'cov'], [tot_run_times, tot_test_times, tot_cov_times]):
            m, q1, q3 = compute_statistics(times)
            tmp[key] = (m, q3 - q1)
        results[case] = tmp
    return results