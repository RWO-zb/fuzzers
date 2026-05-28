import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from collections import Counter
import os

LOG_FILE = 'selection_log.pkl'
OBS_FILE = 'obs_sequences.pkl'
PERF_FILE = 'perf_meta.pkl' 
PLOT_3_FILE = 'MountainCar_crash_generation_histogram.png'
PLOT_4_FILE = 'MountainCar_unique_crashes_over_time.png'       
PLOT_6_FILE = 'MountainCar_survival_steps_boxplot.png'
PLOT_RQ2_FILE = 'MountainCar_RQ2_diversity_metrics.png'

# Set to True when you want to generate figures.
# When False, all metrics are still calculated and printed, but no plots are saved.
ENABLE_PLOTS = False
RQ2_ONLY = True

# RQ2 metrics are calculated on fuzzing-stage cases only.
# Set to None to use all fuzzing-stage cases.
RQ2_FUZZ_SAMPLE_LIMIT = 5000
KMEANS_N_INIT = 3

# Theoretical state space size based on 50x50 grid partition
THEORETICAL_STATE_SPACE = 50 * 50 
GRID_SIZE = (50, 50)

RANGES = {
    'state_pos': (-1.2, 0.6),
    'state_vel': (-0.07, 0.07),
    'bd_pos': (-1.2, 0.6),
    'bd_speed': (0.0, 0.05),
}

def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None

def merge_and_deduplicate(logs, obs_seqs):
    """
    Merge selection_log and obs_sequences, and perform strict deduplication using raw state data.
    Maintains priority overriding: a new Crash record overrides an old safe record.
    """
    if len(logs) != len(obs_seqs):
        print(f"Warning: Logs count ({len(logs)}) and Obs count ({len(obs_seqs)}) mismatch. Truncating.")
    
    min_len = min(len(logs), len(obs_seqs))
    state_to_entry = {}
    
    for i in range(min_len):
        entry = logs[i]
        traj = obs_seqs[i]
        state = entry.get('mutate_state')
        
        if state is None: continue
            
        # Use the original float array as a tuple for strict deduplication hash key
        state_key = tuple(state)
        
        entry_copy = entry.copy()
        entry_copy['output_trajectory'] = traj
        entry_copy['survival_steps'] = len(traj)
        
        if state_key not in state_to_entry:
            state_to_entry[state_key] = entry_copy
        else:
            old_entry = state_to_entry[state_key]
            if entry_copy.get('did_crash', False) and not old_entry.get('did_crash', False):
                state_to_entry[state_key] = entry_copy

    return list(state_to_entry.values())

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

    positions = seq_arr[:, 0]
    velocities = seq_arr[:, 1]
    return np.max(positions), np.mean(np.abs(velocities))

def is_fuzz_stage_entry(log_entry):
    return (
        isinstance(log_entry, dict)
        and 'seed_state' in log_entry
        and 'mutate_state' in log_entry
    )

def calculate_rq2_trends(logs, obs_seqs, grid_size=GRID_SIZE, max_fuzz_cases=None):
    """
    Calculate RQ2-style cumulative metrics over the original fuzzing episode order.

    State coverage counts visited position/velocity grid cells across all trajectories.
    Behavior diversity counts unique behavior descriptors: max position x average speed.
    Fault diversity counts unique behavior descriptors among crashing episodes only.
    Set max_fuzz_cases to limit the calculation to the first N fuzzing-stage cases.
    """
    if not logs or not obs_seqs:
        print("No data loaded for RQ2 diversity metrics.")
        return None

    visited_state_bins = set()
    visited_behavior_bins = set()
    visited_fault_bins = set()
    history = {
        'episodes': [],
        'state_coverage': [],
        'behavior_diversity': [],
        'fault_diversity': [],
        'unique_crash_source_seeds': [],
        'fault_mean_ttd': [],
        'crash_source_mean_ttd': [],
    }

    if len(logs) != len(obs_seqs):
        print(f"Warning: RQ2 logs count ({len(logs)}) and obs count ({len(obs_seqs)}) mismatch. Truncating.")

    min_len = min(len(logs), len(obs_seqs))
    skipped_non_fuzz = 0
    fuzz_cases_processed = 0
    crash_source_seed_ids = set()
    fault_first_seen_times = {}
    crash_source_first_seen_times = {}

    for i in range(min_len):
        sequence = np.asarray(obs_seqs[i])
        log_entry = logs[i]
        if not is_fuzz_stage_entry(log_entry):
            skipped_non_fuzz += 1
            continue

        if max_fuzz_cases is not None and fuzz_cases_processed >= max_fuzz_cases:
            break

        is_crash = log_entry.get('did_crash', False)

        if sequence.ndim == 1 and sequence.size >= 2:
            sequence = sequence.reshape(-1, 2)

        for state in sequence:
            if len(state) >= 2:
                state_idx = get_grid_index(
                    (state[0], state[1]),
                    (RANGES['state_pos'], RANGES['state_vel']),
                    grid_size,
                )
                visited_state_bins.add(state_idx)

        bd_values = calc_behavior_descriptor(sequence)
        bd_idx = get_grid_index(
            bd_values,
            (RANGES['bd_pos'], RANGES['bd_speed']),
            grid_size,
        )
        visited_behavior_bins.add(bd_idx)

        if is_crash:
            visited_fault_bins.add(bd_idx)
            event_time = log_entry.get('crash_time')
            event_time = float(event_time) if event_time is not None else 0.0
            if bd_idx not in fault_first_seen_times:
                fault_first_seen_times[bd_idx] = event_time

            seed_id = log_entry.get('root_id')
            if seed_id is None:
                seed_state = log_entry.get('seed_state')
                if seed_state is not None:
                    seed_id = tuple(np.asarray(seed_state).flatten())
            if seed_id is not None:
                crash_source_seed_ids.add(seed_id)
                if seed_id not in crash_source_first_seen_times:
                    crash_source_first_seen_times[seed_id] = event_time

        fuzz_cases_processed += 1
        history['episodes'].append(fuzz_cases_processed)
        history['state_coverage'].append(len(visited_state_bins))
        history['behavior_diversity'].append(len(visited_behavior_bins))
        history['fault_diversity'].append(len(visited_fault_bins))
        history['unique_crash_source_seeds'].append(len(crash_source_seed_ids))
        history['fault_mean_ttd'].append(np.mean(list(fault_first_seen_times.values())) if fault_first_seen_times else 0.0)
        history['crash_source_mean_ttd'].append(np.mean(list(crash_source_first_seen_times.values())) if crash_source_first_seen_times else 0.0)

    if skipped_non_fuzz:
        print(f"Skipped {skipped_non_fuzz} non-fuzz-stage entries while calculating RQ2 metrics.")

    return history

def truncate_rq2_history(history, limit):
    if history is None or limit is None:
        return history
    return {key: values[:limit] for key, values in history.items()}

def calculate_rq2_metric_sets(logs, obs_seqs):
    total_history = calculate_rq2_trends(logs, obs_seqs, max_fuzz_cases=None)
    limited_history = truncate_rq2_history(total_history, RQ2_FUZZ_SAMPLE_LIMIT)
    return {
        'limited': limited_history,
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

def print_single_rq2_metrics(history, label):
    if not history or not history['episodes']:
        print(f"  {label}: no RQ2 diversity data to report.")
        return False

    auc_metrics = calculate_auc_metrics(history)
    print(f"  [{label}]")
    print(f"    State Coverage:     {history['state_coverage'][-1]} grid bins")
    print(f"    Behavior Diversity: {history['behavior_diversity'][-1]} behavior bins")
    print(f"    Behavior Diversity AUC:      {auc_metrics['behavior_auc']:.4f}")
    print(f"    Behavior Diversity Mean AUC: {auc_metrics['behavior_mean_auc']:.4f}")
    print(f"    Fault Diversity:    {history['fault_diversity'][-1]} fault bins")
    print(f"    Fault Diversity AUC:         {auc_metrics['fault_auc']:.4f}")
    print(f"    Fault Diversity Mean AUC:    {auc_metrics['fault_mean_auc']:.4f}")
    print(f"    Fault Diversity Mean TTD:    {history['fault_mean_ttd'][-1]:.4f} sec")
    print(f"    Crash Source Seeds: {history['unique_crash_source_seeds'][-1]}")
    print(f"    Crash Source Seeds AUC:      {auc_metrics['crash_source_auc']:.4f}")
    print(f"    Crash Source Seeds Mean AUC: {auc_metrics['crash_source_mean_auc']:.4f}")
    print(f"    Crash Source Seeds Mean TTD: {history['crash_source_mean_ttd'][-1]:.4f} sec")
    return True

def print_rq2_metrics(history):
    if isinstance(history, dict) and 'limited' in history and 'total' in history:
        print("\n[0. RQ2 Cumulative Diversity Metrics]")
        limit_label = (
            f"First {RQ2_FUZZ_SAMPLE_LIMIT} Fuzz Cases"
            if RQ2_FUZZ_SAMPLE_LIMIT is not None
            else "Limited Range Disabled"
        )
        printed_limited = print_single_rq2_metrics(history['limited'], limit_label)
        printed_total = print_single_rq2_metrics(history['total'], "All Fuzz Cases")
        print()
        return printed_limited or printed_total

    if not history or not history['episodes']:
        print("No RQ2 diversity data to report.")
        return False

    print("\n[0. RQ2 Cumulative Diversity Metrics]")
    print(f"  State Coverage:     {history['state_coverage'][-1]} grid bins")
    print(f"  Behavior Diversity: {history['behavior_diversity'][-1]} behavior bins")
    print(f"  Fault Diversity:    {history['fault_diversity'][-1]} fault bins")
    print(f"  Crash Source Seeds: {history['unique_crash_source_seeds'][-1]}")
    print()
    return True

def plot_rq2_metrics(history, save_path=PLOT_RQ2_FILE, enable_plot=ENABLE_PLOTS):
    if not print_rq2_metrics(history):
        return

    if not enable_plot:
        print("Plotting disabled. Set ENABLE_PLOTS = True to save figures.\n")
        return

    if isinstance(history, dict) and 'limited' in history and 'total' in history:
        history = history['limited']

    episodes = history['episodes']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_config = [
        {
            'key': 'state_coverage',
            'title': 'State Coverage',
            'ylabel': '# Unique State Bins',
            'color': '#1f77b4',
            'desc': 'Grid: Position x Velocity',
        },
        {
            'key': 'behavior_diversity',
            'title': 'Behavior Diversity',
            'ylabel': '# Unique Behavior Bins',
            'color': '#2ca02c',
            'desc': 'Grid: MaxPos x AvgSpeed',
        },
        {
            'key': 'fault_diversity',
            'title': 'Fault Diversity',
            'ylabel': '# Unique Fault Bins',
            'color': '#d62728',
            'desc': 'Crashing episodes only',
        },
    ]

    for ax, config in zip(axes, metrics_config):
        data = history[config['key']]
        ax.plot(episodes, data, color=config['color'], linewidth=2.5, label='CureFuzz')
        ax.fill_between(episodes, data, color=config['color'], alpha=0.1)
        ax.set_title(config['title'], fontweight='bold', pad=12)
        ax.set_xlabel('Episodes')
        ax.set_ylabel(config['ylabel'])
        ax.text(
            0.05,
            0.95,
            config['desc'],
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )
        ax.set_ylim(bottom=0)
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"RQ2 diversity plot saved to {save_path}\n")

def analyze_and_plot_comprehensive_metrics(original_log, deduplicated_log, perf_data=None):
    print(f"\n{'='*85}")
    print(f"{'Academic-Grade Crash & Diversity Analysis (Strictly did_crash == True)':^85}")
    print(f"{'='*85}")
    
    # --- 1. Global Fuzzing Metrics (Overhead, Hit Ratio, Coverage) ---
    total_mutations = len(original_log)
    total_valid_crashes = sum(1 for e in original_log if e.get('did_crash', False))
    hit_ratio = (total_valid_crashes / total_mutations * 100) if total_mutations > 0 else 0
    
    explored_unique_states = len(deduplicated_log)
    state_space_coverage = (explored_unique_states / THEORETICAL_STATE_SPACE) * 100
    
    print("[1. Overhead, Hit Ratio & State Space Coverage]")
    print(f"  Total Mutations Executed:   {total_mutations}")
    print(f"  Valid Crash Mutations:      {total_valid_crashes}")
    print(f"  Hit Ratio (Valid Rate):     {hit_ratio:.2f}%  <-- % of mutations leading to a crash")
    
    if perf_data:
        total_t = perf_data['total_wall_time']
        algo_t = perf_data['algo_logic_time']
        overhead_ratio = (algo_t / total_t) * 100 if total_t > 0 else 0
    else:
        print()

    # --- 2. Data Extraction for Crash Analysis ---
    inputs, outputs, times, depths, raw_survival_steps = [], [], [], [], []
    
    for entry in deduplicated_log:
        if entry.get('did_crash', False) == True:
            inputs.append(entry['mutate_state'])
            outputs.append(np.array(entry['output_trajectory']).flatten())
            depths.append(entry.get('parent_depth', 0) + 1)
            raw_survival_steps.append(entry['survival_steps'])
            
            t = entry.get('crash_time', 0.0)
            times.append(t if t is not None else 0.0)
                
    unique_crash_count = len(inputs)
    if unique_crash_count < 2:
        print(f"Not enough crash data to calculate metrics (Found {unique_crash_count}, needs >= 2).")
        return
        
    inputs = np.array(inputs)
    times = np.array(times)
    depths = np.array(depths)
    raw_survival_steps = np.array(raw_survival_steps)
    
    times_hrs = np.sort(times / 3600.0)
    max_time_hrs = max([e.get('crash_time', 0.0) for e in original_log if e.get('crash_time') is not None] + [0.0]) / 3600.0
    if max_time_hrs <= 0: max_time_hrs = times_hrs[-1] if len(times_hrs) > 0 else 1.0

    # --- 3. Basic Efficiency & Survival Depth Analysis ---
    intervals_hrs = np.diff(np.insert(times_hrs, 0, 0.0))
    mean_interval = np.mean(intervals_hrs)      
    median_interval = np.median(intervals_hrs)  
    
    print("[2. Basic Crash Efficiency & Episode Depth]")
    print(f"  Total Unique Crashes Discovered: {unique_crash_count}")
    print(f"  Mean Interval per Crash:         {mean_interval:.4f} hours (~{mean_interval*3600:.1f} sec)")
    print(f"  Median Interval per Crash:       {median_interval:.4f} hours (~{median_interval*3600:.1f} sec)")
    print()
    
    if ENABLE_PLOTS:
        cumulative_crashes = np.arange(1, len(times_hrs) + 1)
        plt.figure(figsize=(12, 7))
        plt.step(times_hrs, cumulative_crashes, where='post', color='darkred', linewidth=2, label='Unique Crash Inputs')
        plt.fill_between(times_hrs, cumulative_crashes, step='post', color='darkred', alpha=0.1)
        plt.title('MountainCar: Cumulative Unique Crashes Discovered Over Time')
        plt.xlabel('Time Elapsed (hours)')
        plt.ylabel('Number of Unique Crashing Inputs')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(left=0, right=max_time_hrs)
        plt.ylim(bottom=0)
        plt.legend()
        plt.savefig(PLOT_4_FILE)
        plt.close()

    # --- 4. Trajectory Padding for Sequence Clustering ---
    max_len = max(len(t) for t in outputs)
    padded_outputs = [np.pad(t, (0, max_len - len(t)), mode='constant') for t in outputs]
    outputs_padded = np.array(padded_outputs)
    
    # --- 5. Advanced Diversity Quality Metrics (PCA + KMeans) ---
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
            best_score = silhouette_score(
                reduced_data,
                labels,
                sample_size=min(5000, n_samples),
                random_state=42
            )
            
            for k in range(3, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=KMEANS_N_INIT)
                labels = kmeans.fit_predict(reduced_data)
                score = silhouette_score(
                    reduced_data,
                    labels,
                    sample_size=min(5000, n_samples),
                    random_state=42
                )
                if score >= best_score * 1.20:
                    best_score = score
                    best_k = k
                    
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=KMEANS_N_INIT)
        labels = kmeans.fit_predict(reduced_data)
        
        # Cluster Distance & Entropy Calculation
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
        
        # Time-To-Discovery (TTD) & AUC Calculation
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

        # Boxplot Generation
        if ENABLE_PLOTS and name == "Output" and raw_lengths is not None:
            cluster_steps = [raw_lengths[labels == k] for k in range(best_k)]
            plt.figure(figsize=(10, 6))
            plt.boxplot(cluster_steps, tick_labels=[f"Cluster {k+1}\n(n={len(cluster_steps[k])})" for k in range(best_k)])
            plt.title('MountainCar: Crash Episode Length (Survival Steps) per Fault Type')
            plt.ylabel('Timesteps until Crash')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.savefig(PLOT_6_FILE)
            plt.close()

    compute_diversity_metrics(inputs, times, "Input")
    compute_diversity_metrics(outputs_padded, times, "Output", raw_lengths=raw_survival_steps)
    print(f"{'='*85}\n")

def plot_generation_histogram(deduplicated_log):
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
    print(f"  Deepest Crash Found at Generation: {max_gen}")

    if ENABLE_PLOTS:
        generation_counts = Counter(crash_generations)
        generations = range(0, max_gen + 2)
        counts = [generation_counts.get(gen, 0) for gen in generations]
        plt.figure(figsize=(12, 7))
        plt.bar(generations, counts, color='red', alpha=0.7, zorder=3)
        plt.title('MountainCar: Histogram of Unique Crash Generations')
        plt.xlabel('Mutation Generation')
        plt.ylabel('Number of Unique Crashing Inputs')
        step = max(1, (max_gen // 20))
        plt.xticks(np.arange(0, max_gen + 2, step=step))
        plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
        plt.savefig(PLOT_3_FILE)
        plt.close()

def main():
    original_log_data = load_data(LOG_FILE)
    obs_seqs = load_data(OBS_FILE)
    
    if not original_log_data or not obs_seqs: 
        print("Failed to load log or observation data.")
        return

    rq2_history = calculate_rq2_metric_sets(original_log_data, obs_seqs)
    plot_rq2_metrics(rq2_history)
    if RQ2_ONLY:
        print("RQ2-only analysis completed.")
        return
        
    perf_data = load_data(PERF_FILE) 
    deduplicated_log = merge_and_deduplicate(original_log_data, obs_seqs)
    if not deduplicated_log: 
        print("Deduplicated log is empty.")
        return
    
    analyze_and_plot_comprehensive_metrics(original_log_data, deduplicated_log, perf_data)
    plot_generation_histogram(deduplicated_log)

    if ENABLE_PLOTS:
        print("All analysis and plotting completed. Check the generated PNG files.")
    else:
        print("All analysis completed. Plotting is disabled.")

if __name__ == "__main__":
    main()
