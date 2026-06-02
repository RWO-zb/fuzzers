import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from collections import Counter
import os

# Configuration
# Example (MDPFuzz): 'logs/MC_DQN_NoCov_5_0.01_0.1_0_12h'
# Example (RT):      'logs/MC_DQN_RT_1022_10000it'
BASE_PREFIX = 'MC_DQN_RT_1022_12h'  # <-- Set this to your log prefix (without _logs.txt or _obs.txt)

# Infer method from prefix
IS_RT = '_RT_' in BASE_PREFIX
METHOD_NAME = 'Random Testing' if IS_RT else 'MDPFuzz'

LOG_FILE = f"{BASE_PREFIX}_logs.txt"
OBS_FILE = f"{BASE_PREFIX}_obs.txt"

PLOT_CUMULATIVE_FILE = f"{METHOD_NAME.replace(' ', '_')}_unique_crashes_over_time.png"       
PLOT_GEN_FILE = f"{METHOD_NAME.replace(' ', '_')}_crash_generation_histogram.png"
PLOT_SURVIVAL_FILE = f"{METHOD_NAME.replace(' ', '_')}_survival_steps_boxplot.png"

ENABLE_PLOTS = False
RQ2_ONLY = True
RQ2_FUZZ_SAMPLE_LIMIT = 5000
KMEANS_N_INIT = 3
GRID_SIZE = (50, 50)
RANGES = {
    'state_pos': (-1.2, 0.6),
    'state_vel': (-0.07, 0.07),
    'bd_pos': (-1.2, 0.6),
    'bd_speed': (0.0, 0.05),
}

# MountainCar theoretical state space grid (50x50)
THEORETICAL_STATE_SPACE = 50 * 50 

def parse_optional_float(value):
    if value in (None, 'None', ''):
        return None
    return float(value)

def resolve_event_time(run_time, crash_time, start_time):
    if crash_time is not None:
        if start_time and crash_time >= start_time:
            return max(0.0, crash_time - start_time)
        return max(0.0, crash_time)
    if run_time is None:
        return 0.0
    if start_time and run_time >= start_time:
        return max(0.0, run_time - start_time)
    return max(0.0, run_time)

def get_grid_index(values, ranges, grid_size):
    indices = []
    for val, (min_val, max_val), bins in zip(values, ranges, grid_size):
        norm = (val - min_val) / (max_val - min_val) if max_val != min_val else 0
        idx = int(norm * bins)
        indices.append(int(np.clip(idx, 0, bins - 1)))
    return tuple(indices)

def calc_behavior_descriptor(sequence):
    seq_arr = np.asarray(sequence)
    if len(seq_arr) == 0:
        return -1.2, 0.0
    if seq_arr.ndim == 1:
        seq_arr = seq_arr.reshape(-1, 2)
    return np.max(seq_arr[:, 0]), np.mean(np.abs(seq_arr[:, 1]))

def load_mdpfuzz_rq2_data(log_file, obs_file):
    if not os.path.exists(log_file) or not os.path.exists(obs_file):
        return [], []

    log_rows = []
    with open(log_file, 'r') as f:
        headers = [h.strip() for h in f.readline().strip().split('; ')]
        for line in f:
            vals = [v.strip() for v in line.strip().split('; ')]
            if len(vals) >= len(headers):
                log_rows.append(dict(zip(headers, vals)))

    obs_rows = []
    with open(obs_file, 'r') as f:
        current_info = None
        current_traj = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("--- Test Case Info:"):
                if current_info is not None:
                    gen = current_info.get('Generation', 0)
                    if IS_RT or gen != 0:
                        obs_rows.append((current_info, np.array(current_traj)))
                json_str = line[len("--- Test Case Info: "):-len(" ---")]
                current_info = json.loads(json_str)
                current_traj = []
            else:
                if current_info is not None:
                    try:
                        vals = [float(x) for x in line.split(',')]
                        if len(vals) >= 2:
                            current_traj.append(vals[:2])
                    except ValueError:
                        continue
        if current_info is not None:
            gen = current_info.get('Generation', 0)
            if IS_RT or gen != 0:
                obs_rows.append((current_info, np.array(current_traj)))

    valid_logs = []
    for row in log_rows:
        try:
            gen = int(float(row.get('Generation', 0)))
        except ValueError:
            gen = 0
        if IS_RT or gen != 0:
            valid_logs.append(row)

    fuzz_start_time = None
    if valid_logs:
        fuzz_start_time = parse_optional_float(valid_logs[0].get('RunTime', 'None'))

    min_len = min(len(obs_rows), len(valid_logs))
    data = []
    for i in range(min_len):
        info, traj = obs_rows[i]
        log_row = valid_logs[i]
        seed_id = log_row.get('SeedID')
        if seed_id in (None, 'None', ''):
            seed_id = info.get('SeedID')
        run_time = parse_optional_float(log_row.get('RunTime', 'None'))
        crash_time = parse_optional_float(log_row.get('CrashTime', 'None'))
        data.append({
            'sequence': traj,
            'is_crash': bool(info.get('Oracle', False)),
            'seed_id': seed_id,
            'event_time': resolve_event_time(run_time, crash_time, fuzz_start_time),
        })
    return data, obs_rows

def calculate_rq2_trends(rq2_data, max_fuzz_cases=None):
    visited_state_bins = set()
    visited_behavior_bins = set()
    visited_fault_bins = set()
    crash_source_seed_ids = set()
    history = {
        'episodes': [],
        'state_coverage': [],
        'behavior_diversity': [],
        'fault_diversity': [],
        'unique_crash_source_seeds': [],
        'fault_mean_ttd': [],
        'crash_source_mean_ttd': [],
        'event_times': [],
    }
    fault_first_seen_times = {}
    crash_source_first_seen_times = {}

    for item in rq2_data:
        if max_fuzz_cases is not None and len(history['episodes']) >= max_fuzz_cases:
            break

        sequence = item['sequence']
        for state in sequence:
            if len(state) >= 2:
                visited_state_bins.add(get_grid_index(
                    (state[0], state[1]),
                    (RANGES['state_pos'], RANGES['state_vel']),
                    GRID_SIZE,
                ))

        bd_idx = get_grid_index(
            calc_behavior_descriptor(sequence),
            (RANGES['bd_pos'], RANGES['bd_speed']),
            GRID_SIZE,
        )
        visited_behavior_bins.add(bd_idx)

        if item['is_crash']:
            visited_fault_bins.add(bd_idx)
            event_time = float(item.get('event_time', 0.0) or 0.0)
            if bd_idx not in fault_first_seen_times:
                fault_first_seen_times[bd_idx] = event_time

            seed_id = item.get('seed_id')
            if seed_id not in (None, 'None', ''):
                crash_source_seed_ids.add(seed_id)
                if seed_id not in crash_source_first_seen_times:
                    crash_source_first_seen_times[seed_id] = event_time

        history['episodes'].append(len(history['episodes']) + 1)
        history['state_coverage'].append(len(visited_state_bins))
        history['behavior_diversity'].append(len(visited_behavior_bins))
        history['fault_diversity'].append(len(visited_fault_bins))
        history['unique_crash_source_seeds'].append(len(crash_source_seed_ids))
        history['fault_mean_ttd'].append(np.mean(list(fault_first_seen_times.values())) if fault_first_seen_times else 0.0)
        history['crash_source_mean_ttd'].append(np.mean(list(crash_source_first_seen_times.values())) if crash_source_first_seen_times else 0.0)
        history['event_times'].append(float(item.get('event_time', 0.0) or 0.0))

    return history

def truncate_rq2_history(history, limit):
    if history is None or limit is None:
        return history
    return {key: values[:limit] for key, values in history.items()}

def calculate_rq2_metric_sets(rq2_data):
    total_history = calculate_rq2_trends(rq2_data, max_fuzz_cases=None)
    return {
        'limited': truncate_rq2_history(total_history, RQ2_FUZZ_SAMPLE_LIMIT),
        'total': total_history,
    }

def calculate_auc_metrics(history):
    if not history or not history['episodes']:
        return {
            'behavior_auc': 0.0,
            'behavior_mean_auc': 0.0,
            'fault_auc': 0.0,
            'fault_mean_auc': 0.0,
            'crash_source_auc': 0.0,
            'crash_source_mean_auc': 0.0,
        }

    episodes = np.asarray(history['episodes'], dtype=float)

    def curve_auc(key):
        values = np.asarray(history[key], dtype=float)
        try:
            auc_value = np.trapezoid(values, episodes)
        except AttributeError:
            auc_value = np.trapz(values, episodes)
        mean_auc = auc_value / episodes[-1] if episodes[-1] > 0 else 0.0
        return auc_value, mean_auc

    behavior_auc, behavior_mean_auc = curve_auc('behavior_diversity')
    fault_auc, fault_mean_auc = curve_auc('fault_diversity')
    crash_source_auc, crash_source_mean_auc = curve_auc('unique_crash_source_seeds')

    return {
        'behavior_auc': behavior_auc,
        'behavior_mean_auc': behavior_mean_auc,
        'fault_auc': fault_auc,
        'fault_mean_auc': fault_mean_auc,
        'crash_source_auc': crash_source_auc,
        'crash_source_mean_auc': crash_source_mean_auc,
    }

def calculate_fault_category_discovery_metrics(history):
    if not history or not history['episodes'] or 'event_times' not in history:
        return {
            'fault_discovery_auc': 0.0,
            'fault_discovery_mean_auc': 0.0,
            'fault_discovery_mean_ttd': 0.0,
        }

    event_times = np.asarray(history['event_times'], dtype=float)
    fault_counts = np.asarray(history['fault_diversity'], dtype=float)
    if len(event_times) == 0:
        return {
            'fault_discovery_auc': 0.0,
            'fault_discovery_mean_auc': 0.0,
            'fault_discovery_mean_ttd': 0.0,
        }

    order = np.argsort(event_times, kind='stable')
    times_hrs = event_times[order] / 3600.0
    counts = fault_counts[order]
    x_steps = [0.0]
    y_steps = [0.0]

    last_count = 0.0
    discovery_times = []
    for t_hr, count in zip(times_hrs, counts):
        if count > last_count:
            x_steps.extend([t_hr, t_hr])
            y_steps.extend([last_count, count])
            discovery_times.extend([t_hr] * int(count - last_count))
            last_count = count

    max_time_hr = float(np.max(times_hrs)) if len(times_hrs) > 0 else 0.0
    if max_time_hr > x_steps[-1]:
        x_steps.append(max_time_hr)
        y_steps.append(last_count)

    try:
        auc_value = np.trapezoid(y_steps, x_steps)
    except AttributeError:
        auc_value = np.trapz(y_steps, x_steps)

    mean_auc = auc_value / max_time_hr if max_time_hr > 0 else 0.0
    mean_ttd_sec = float(np.mean(discovery_times) * 3600.0) if discovery_times else 0.0

    return {
        'fault_discovery_auc': auc_value,
        'fault_discovery_mean_auc': mean_auc,
        'fault_discovery_mean_ttd': mean_ttd_sec,
    }

def print_single_rq2_metrics(history, label):
    if not history or not history['episodes']:
        print(f"  {label}: no RQ2 diversity data to report.")
        return False
    auc_metrics = calculate_auc_metrics(history)
    fault_discovery_metrics = calculate_fault_category_discovery_metrics(history)
    print(f"  [{label}]")
    print(f"    State Coverage:     {history['state_coverage'][-1]} grid bins")
    print(f"    Behavior Diversity: {history['behavior_diversity'][-1]} behavior bins")
    print(f"    Behavior Diversity AUC:      {auc_metrics['behavior_auc']:.4f}")
    print(f"    Behavior Diversity Mean AUC: {auc_metrics['behavior_mean_auc']:.4f}")
    print(f"    Fault Diversity:    {history['fault_diversity'][-1]} fault bins")
    print(f"    Fault Diversity AUC:         {auc_metrics['fault_auc']:.4f}")
    print(f"    Fault Diversity Mean AUC:    {auc_metrics['fault_mean_auc']:.4f}")
    print(f"    Fault Diversity Mean TTD:    {history['fault_mean_ttd'][-1]:.4f} sec")
    print(f"    Fault-category Discovery AUC:      {fault_discovery_metrics['fault_discovery_auc']:.4f} category*hours")
    print(f"    Fault-category Discovery Mean AUC: {fault_discovery_metrics['fault_discovery_mean_auc']:.4f} categories")
    print(f"    Fault-category Discovery TTD:      {fault_discovery_metrics['fault_discovery_mean_ttd']:.4f} sec")
    print(f"    Crash Source Seeds: {history['unique_crash_source_seeds'][-1]}")
    print(f"    Crash Source Seeds AUC:      {auc_metrics['crash_source_auc']:.4f}")
    print(f"    Crash Source Seeds Mean AUC: {auc_metrics['crash_source_mean_auc']:.4f}")
    print(f"    Crash Source Seeds Mean TTD: {history['crash_source_mean_ttd'][-1]:.4f} sec")
    return True

def print_rq2_metrics(metric_sets):
    print("\n[0. RQ2 Cumulative Diversity Metrics]")
    limit_label = f"First {RQ2_FUZZ_SAMPLE_LIMIT} Fuzz Cases" if RQ2_FUZZ_SAMPLE_LIMIT is not None else "Limited Range Disabled"
    printed_limited = print_single_rq2_metrics(metric_sets.get('limited'), limit_label)
    printed_total = print_single_rq2_metrics(metric_sets.get('total'), "All Fuzz Cases")
    print()
    return printed_limited or printed_total

def load_and_merge_mdpfuzz_data(log_file, obs_file):
    """
    Parse and merge _logs.txt and _obs.txt.
    """
    if not os.path.exists(log_file) or not os.path.exists(obs_file):
        print(f"Error: Log or Obs file not found for prefix '{BASE_PREFIX}'.")
        return None, None
        
    print(f"Parsing log files for [{METHOD_NAME}]...")
    
    # Parse logs
    logs = []
    with open(log_file, 'r') as f:
        headers = f.readline().strip().split('; ')
        for line in f:
            line = line.strip()
            if not line: continue
            vals = line.split('; ')
            if len(vals) < len(headers): continue
            logs.append(dict(zip(headers, vals)))
            
    # Parse obs
    obs_data = []
    with open(obs_file, 'r') as f:
        current_info = None
        current_traj = []
        is_crash = False
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("--- Test Case Info:"):
                if current_info is not None:
                    obs_data.append((current_info, current_traj))
                json_str = line[len("--- Test Case Info: "):-len(" ---")]
                current_info = json.loads(json_str)
                current_traj = []
                is_crash = current_info.get('Oracle', False)
            else:
                if is_crash:
                    current_traj.append([float(x) for x in line.split(',')])
        if current_info is not None:
            obs_data.append((current_info, current_traj))
            
    offset = max(0, len(logs) - len(obs_data))
    if offset > 0:
        print(f"  -> Offset detected: Discarding first {offset} logs (initial phase) to strictly compute fuzzing metrics.")
        logs = logs[offset:]
    elif len(obs_data) > len(logs):
        print(f"  -> Warning: Obs count > Logs count. Will truncate obs_data.")
        obs_data = obs_data[:len(logs)]

    fuzz_start_time = None
    if len(logs) > 0:
        fuzz_start_time = parse_optional_float(logs[0].get('RunTime', 'None'))
        
    merged_logs = []
    total_algo_time = 0.0
    max_run_time = 0.0
    
    for i in range(len(logs)):
        log_row = logs[i]
        obs_info, traj = obs_data[i]
        
        mutate_state = np.array(obs_info['Input'])
        did_crash = obs_info.get('Oracle', False)
        parent_depth = obs_info.get('Generation', 0)
        survival_steps = obs_info.get('Steps', len(traj))
        
        # Handle 'None' values in RT and make relative to fuzz_start_time
        run_time = parse_optional_float(log_row.get('RunTime', 'None'))
        relative_run_time = resolve_event_time(run_time, None, fuzz_start_time)
        max_run_time = max(max_run_time, relative_run_time)
        
        crash_time = parse_optional_float(log_row.get('CrashTime', 'None'))
        relative_crash_time = resolve_event_time(run_time, crash_time, fuzz_start_time)
        
        algo_time = log_row.get('CoverageTime', 'None')
        algo_time = float(algo_time) if algo_time != 'None' else 0.0
        total_algo_time += algo_time
        
        # Strictly enforce the 12-hour evaluation boundary for consistency
        if relative_run_time > 12.0 * 3600:
            continue
            
        merged_logs.append({
            'mutate_state': mutate_state,
            'did_crash': did_crash,
            'parent_depth': parent_depth,
            'survival_steps': survival_steps,
            'output_trajectory': np.array(traj) if did_crash else None,
            'run_time': relative_run_time,
            'crash_time': relative_crash_time
        })
        
    perf_data = {
        'total_wall_time': max_run_time,
        'algo_logic_time': total_algo_time
    }
        
    return merged_logs, perf_data

def deduplicate_log(merged_logs):
    """Deduplicate logs with tuple hashing, prioritizing crashes."""
    state_to_entry = {}
    
    for entry in merged_logs:
        state = entry['mutate_state']
        if state is None: continue
        state_key = tuple(state)
        
        if state_key not in state_to_entry:
            state_to_entry[state_key] = entry
        else:
            old_entry = state_to_entry[state_key]
            if entry.get('did_crash', False) and not old_entry.get('did_crash', False):
                state_to_entry[state_key] = entry

    return list(state_to_entry.values())

def plot_cumulative_crashes_from_logs(log_file, is_rt_mode=False):
    """Strictly follows mdpfuzz-RQ1.py logic to plot cumulative unique crashes."""
    MAX_H = 12.0
    VIEW_LIMIT_H = 12.5
    
    unique_crashes = []
    seen_inputs = set()
    fuzz_start_time = None  
    
    with open(log_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader, None)
        if not headers: 
            times = np.array([])
        else:
            headers = [h.strip() for h in headers]
            idx_input = headers.index('Input')
            idx_oracle = headers.index('Oracle')
            idx_runtime = headers.index('RunTime')
            idx_gen = headers.index('Generation') 
            rows = []
            for row in reader:
                if row: rows.append(row)
            rows.sort(key=lambda x: float(x[idx_runtime]) if x[idx_runtime].strip() != 'None' else 0)
            
            for row in rows:
                if len(row) <= idx_gen: continue
                gen_val = int(float(row[idx_gen]))
                if not is_rt_mode and gen_val == 0:
                    continue 
                run_time = float(row[idx_runtime])
                if fuzz_start_time is None:
                    fuzz_start_time = run_time       
                relative_time = run_time - fuzz_start_time   
                if relative_time > MAX_H * 3600:
                    continue
                oracle_str = row[idx_oracle].strip()
                inp_str = row[idx_input].strip() 
                if oracle_str == 'True':
                    if inp_str not in seen_inputs:
                        seen_inputs.add(inp_str)
                        unique_crashes.append(relative_time)            
            
    times = np.array(unique_crashes)
    
    markers_x_h = np.arange(2, MAX_H + 0.1, 2)
    plt.figure(figsize=(10, 6))
    plot_color = '#1f77b4' if IS_RT else '#ff7f0e'

    if len(times) == 0:
        print(f"[{METHOD_NAME}] No crashes found in Fuzzing Phase (Logs).")
        x_plot, y_plot = np.array([0]), np.array([0])
        times_hrs = np.array([])
    else:
        times_hrs = times / 3600.0
        x_plot = np.concatenate(([0], times_hrs))
        y_plot = np.concatenate(([0], np.arange(1, len(times_hrs) + 1)))
        if x_plot[-1] < MAX_H:
            x_plot = np.concatenate((x_plot, [MAX_H]))
            y_plot = np.concatenate((y_plot, [y_plot[-1]]))

    plt.step(x_plot, y_plot, where='post', label=METHOD_NAME, color=plot_color)

    if len(times) > 0:
        marker_y_vals = [np.searchsorted(times_hrs, mx, side='right') for mx in markers_x_h]
        plt.plot(markers_x_h, marker_y_vals, linestyle='none', marker='^', 
                 color=plot_color, markersize=8, markeredgecolor='white', markeredgewidth=1)

    plt.xlim(0, VIEW_LIMIT_H)
    plt.xticks(np.arange(0, 13, 2))
    plt.xlabel("Fuzzing Time (h) ") 
    plt.ylabel("Number of Unique Crashes")
    plt.title(f"Cumulative Unique Crashes ({METHOD_NAME})")
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOT_CUMULATIVE_FILE)
    plt.close()
    
    print(f"\n[Plot] Cumulative Unique Crashes plot drawn successfully using raw logs.")
    print(f"       -> Actual total unique crashes recorded in plot: {len(times)}\n")
def analyze_and_plot_comprehensive_metrics(original_log, deduplicated_log, perf_data):
    print(f"\n{'='*85}")
    print(f"{f'[{METHOD_NAME}] Academic-Grade Evaluation (Strictly did_crash == True)':^85}")
    print(f"{'='*85}")
    
    total_mutations = len(original_log)
    total_valid_crashes = sum(1 for e in original_log if e.get('did_crash', False))
    hit_ratio = (total_valid_crashes / total_mutations * 100) if total_mutations > 0 else 0
    
    explored_unique_states = len(deduplicated_log)
    state_space_coverage = (explored_unique_states / THEORETICAL_STATE_SPACE) * 100
    
    print("[1. Global Metrics & State Space Coverage]")
    print(f"  Total Evaluations Executed: {total_mutations}")
    print(f"  Valid Crash Inputs:         {total_valid_crashes}")
    print(f"  Hit Ratio (Valid Rate):     {hit_ratio:.2f}%")
    
    if IS_RT:
        print()
    elif perf_data:
        total_t = perf_data['total_wall_time']
        algo_t = perf_data['algo_logic_time']
        overhead_ratio = (algo_t / total_t) * 100 if total_t > 0 else 0

    inputs, outputs, times, depths, raw_survival_steps = [], [], [], [], []
    
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            inputs.append(entry['mutate_state'])
            outputs.append(np.array(entry['output_trajectory']).flatten())
            depths.append(entry.get('parent_depth', 0) + 1)
            raw_survival_steps.append(entry['survival_steps'])
            
            t = entry.get('crash_time', 0.0)
            times.append(t if t is not None else 0.0)
                
    unique_crash_count = len(inputs)
    if unique_crash_count < 2:
        print(f"Not enough crash data to calculate advanced metrics (Found {unique_crash_count}, needs >= 2).")
        return
        
    inputs = np.array(inputs)
    times = np.array(times)
    depths = np.array(depths)
    raw_survival_steps = np.array(raw_survival_steps)
    
    times_hrs = np.sort(times / 3600.0)
    max_time_hrs = max([e.get('crash_time', 0.0) for e in original_log if e.get('crash_time') is not None] + [0.0]) / 3600.0
    if max_time_hrs <= 0: max_time_hrs = times_hrs[-1] if len(times_hrs) > 0 else 1.0

    intervals_hrs = np.diff(np.insert(times_hrs, 0, 0.0))
    mean_interval = np.mean(intervals_hrs)      
    median_interval = np.median(intervals_hrs)  
    
    print("[2. Basic Crash Efficiency & Episode Depth]")
    print(f"  Total Unique Crashes Discovered: {unique_crash_count}")
    print(f"  Mean Interval per Crash:         {mean_interval:.4f} hours (~{mean_interval*3600:.1f} sec)")
    print(f"  Median Interval per Crash:       {median_interval:.4f} hours (~{median_interval*3600:.1f} sec)")
    print()
    
    # Draw Cumulative unique crashes over time using strict RQ1 logic (from raw logs)
    if ENABLE_PLOTS:
        plot_cumulative_crashes_from_logs(LOG_FILE, IS_RT)

    # Trajectory Padding
    max_len = max(len(t) for t in outputs)
    padded_outputs = [np.pad(t, (0, max_len - len(t)), mode='constant') for t in outputs]
    outputs_padded = np.array(padded_outputs)
    
    def compute_diversity_metrics(data_matrix, times_array, name, raw_lengths=None):
        n_samples = data_matrix.shape[0]
        n_components = min(n_samples, data_matrix.shape[1], 10) 
        pca = PCA(n_components=n_components, random_state=42)
        reduced_data = pca.fit_transform(data_matrix)
        
        best_k = 1
        best_score = -1
        max_k = min(20, n_samples - 1) 
        
        if max_k >= 2:
            best_k = 2
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=KMEANS_N_INIT)
            labels = kmeans.fit_predict(reduced_data)
            # Sample for silhouette score if data is large
            best_score = silhouette_score(reduced_data, labels, sample_size=min(5000, n_samples), random_state=42)
            
            for k in range(3, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=KMEANS_N_INIT)
                labels = kmeans.fit_predict(reduced_data)
                score = silhouette_score(reduced_data, labels, sample_size=min(5000, n_samples), random_state=42)
                if score >= best_score * 1.20:
                    best_score = score
                    best_k = k
                    
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=KMEANS_N_INIT)
        labels = kmeans.fit_predict(reduced_data)
        
        centroids = kmeans.cluster_centers_
        intra_dists = []
        for i in range(best_k):
            cluster_points = reduced_data[labels == i]
            if len(cluster_points) > 0:
                dist = np.mean(cdist(cluster_points, [centroids[i]]))
                intra_dists.append(dist)
        avg_intra_dist = np.mean(intra_dists) if intra_dists else 0.0
        
        if best_k > 1:
            avg_inter_dist = np.mean(pdist(centroids, metric='euclidean'))
        else:
            avg_inter_dist = 0.0
            
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        
        discovery_times_hrs = []
        for cluster_id in range(best_k):
            cluster_times = times_array[labels == cluster_id] / 3600.0
            if len(cluster_times) > 0:
                discovery_times_hrs.append(np.min(cluster_times))
                
        discovery_times_hrs.sort()
        
        x_steps = [0.0]
        y_steps = [0]
        for i, t_hr in enumerate(discovery_times_hrs):
            x_steps.extend([t_hr, t_hr])
            y_steps.extend([y_steps[-1], i + 1])
            
        if max_time_hrs > x_steps[-1]:
            x_steps.append(max_time_hrs)
            y_steps.append(best_k)
            
        try:
            auc_val = np.trapezoid(y_steps, x_steps)
        except AttributeError:
            auc_val = np.trapz(y_steps, x_steps)
            
        mean_ttd = np.mean(discovery_times_hrs) if discovery_times_hrs else 0.0
        
        print(f"[3. {name} Diversity Quality & Efficiency]")
        print(f"  Clusters Discovered (K*):            {best_k}")
        print(f"  Absolute Silhouette Score:           {best_score:.4f} (Range [-1, 1])")
        print(f"  Avg Intra-Cluster Dist (Variance):   {avg_intra_dist:.4f}")
        print(f"  Avg Inter-Cluster Dist (Separation): {avg_inter_dist:.4f}")
        print(f"  Entropy (Distribution Evenness):     {entropy:.4f}")
        print(f"  Mean Time-to-Discovery per Category: {mean_ttd:.4f} hours")
        print(f"  Diversity AUC (Clusters vs Time):    {auc_val:.4f} (category*hours)\n")

        # Survival steps boxplot for Output
        if ENABLE_PLOTS and name == "Output" and raw_lengths is not None:
            cluster_steps = [raw_lengths[labels == k] for k in range(best_k)]
            plt.figure(figsize=(10, 6))
            plt.boxplot(cluster_steps, tick_labels=[f"Cluster {k+1}\n(n={len(cluster_steps[k])})" for k in range(best_k)])
            plt.title(f'Crash Episode Length (Survival Steps) per Fault Type ({METHOD_NAME})')
            plt.ylabel('Timesteps until Crash')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.savefig(PLOT_SURVIVAL_FILE)
            plt.close()

    compute_diversity_metrics(inputs, times, "Input")
    compute_diversity_metrics(outputs_padded, times, "Output", raw_lengths=raw_survival_steps)
    print(f"{'='*85}\n")

def plot_generation_histogram(deduplicated_log):
    if IS_RT:
        print(f"[Evolutionary Depth Analysis]")
        print("  Average/Deepest Generation: N/A (Random Testing operates solely on Generation 0)\n")
        return

    crash_generations = []
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            parent_depth = entry.get('parent_depth')
            if parent_depth is not None:
                crash_generations.append(parent_depth + 1)
            
    if not crash_generations: return

    avg_gen = np.mean(crash_generations)
    median_gen = np.median(crash_generations)
    max_gen = np.max(crash_generations)
    
    print(f"[Evolutionary Depth Analysis]")
    print(f"  Average Crash Generation (Mean):   {avg_gen:.2f}")
    print(f"  Median Crash Generation (Median):  {median_gen:.2f}")
    print(f"  Deepest Crash Found at Generation: {max_gen}\n")

    if ENABLE_PLOTS:
        generation_counts = Counter(crash_generations)
        generations = range(1, max_gen + 2)
        counts = [generation_counts.get(gen, 0) for gen in generations]
        plt.figure(figsize=(10, 6))
        plt.bar(generations, counts, color='#ff7f0e', alpha=0.8, edgecolor='black', zorder=3)
        plt.title('Histogram of Unique Crash Generations (MDPFuzz)')
        plt.xlabel('Mutation Generation')
        plt.ylabel('Number of Unique Crashing Inputs')
        step = max(1, (max_gen // 10))
        plt.xticks(np.arange(1, max_gen + 2, step=step))
        plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
        plt.tight_layout()
        plt.savefig(PLOT_GEN_FILE)
        plt.close()

def main():
    print(f"Loading files for prefix: {BASE_PREFIX}")
    rq2_data, _ = load_mdpfuzz_rq2_data(LOG_FILE, OBS_FILE)
    print_rq2_metrics(calculate_rq2_metric_sets(rq2_data))
    if RQ2_ONLY:
        print(f"RQ2-only analysis completed for [{METHOD_NAME}].")
        return

    merged_logs, perf_data = load_and_merge_mdpfuzz_data(LOG_FILE, OBS_FILE)
    
    if not merged_logs:
        return
        
    deduplicated_log = deduplicate_log(merged_logs)
    if not deduplicated_log: 
        print("Deduplicated log is empty.")
        return
    
    analyze_and_plot_comprehensive_metrics(merged_logs, deduplicated_log, perf_data)
    plot_generation_histogram(deduplicated_log)

    if ENABLE_PLOTS:
        print(f"All analysis and plotting completed for [{METHOD_NAME}]. Check the generated PNG files.")
    else:
        print(f"All analysis completed for [{METHOD_NAME}]. Plotting is disabled.")

if __name__ == "__main__":
    main()
