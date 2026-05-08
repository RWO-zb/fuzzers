import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from collections import Counter
import os
import ast

# Configuration
# Change this prefix to point to the desired log files (e.g., 'results/mc_test' or 'seed42/mc_test')
BASE_PREFIX = 'mc_test'

METHOD_NAME = 'QDFuzz'

CSV_FILE = f"{BASE_PREFIX}_data.csv"
OBS_FILE = f"{BASE_PREFIX}_obs.txt"

PLOT_CUMULATIVE_FILE = f"{METHOD_NAME.replace(' ', '_')}_unique_crashes_over_time.png"       
PLOT_GEN_FILE = f"{METHOD_NAME.replace(' ', '_')}_crash_generation_histogram.png"
PLOT_SURVIVAL_FILE = f"{METHOD_NAME.replace(' ', '_')}_survival_steps_boxplot.png"

# MountainCar theoretical state space grid (50x50)
THEORETICAL_STATE_SPACE = 50 * 50 

def load_and_merge_qdfuzz_data(csv_file, obs_file):
    """
    Parse and merge _data.csv and _obs.txt.
    """
    if not os.path.exists(csv_file) or not os.path.exists(obs_file):
        print(f"Error: CSV or Obs file not found for prefix '{BASE_PREFIX}'.")
        return None, None
        
    print(f"Parsing log files for [{METHOD_NAME}]...")
    
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
            
    # Parse csv
    df = pd.read_csv(csv_file)
    df = df.sort_values(by='discovery_time').reset_index(drop=True)
    
    if len(obs_data) != len(df):
        print(f"  -> Warning: Obs count ({len(obs_data)}) != CSV count ({len(df)}). Will truncate to min.")
        min_len = min(len(obs_data), len(df))
        obs_data = obs_data[:min_len]
        df = df.iloc[:min_len]
        
    # Find fuzz_start_time
    init_df = df[df['mutation_count'] == 0]
    if not init_df.empty:
        fuzz_start_time = init_df['discovery_time'].max()
        print(f"  -> Initialization phase ends at {fuzz_start_time:.2f}s. Discarding init data for metrics.")
    else:
        fuzz_start_time = 0.0

    merged_logs = []
    max_run_time = 0.0
    
    for i in range(len(df)):
        row = df.iloc[i]
        obs_info, traj = obs_data[i]
        
        mut_count = int(row['mutation_count'])
        if mut_count == 0:
            continue # Discard init phase (strictly fuzzing)
            
        inp = row['input']
        if isinstance(inp, str):
            try:
                mutate_state = np.array(json.loads(inp))
            except:
                try:
                    mutate_state = np.array(ast.literal_eval(inp))
                except:
                    mutate_state = inp
        else:
            mutate_state = np.array(inp)
            
        did_crash = bool(row['is_faulty'])
        parent_depth = mut_count
        survival_steps = obs_info.get('Steps', len(traj))
        
        raw_time = float(row['discovery_time'])
        relative_run_time = max(0.0, raw_time - fuzz_start_time)
        max_run_time = max(max_run_time, relative_run_time)
        
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
            'crash_time': relative_run_time if did_crash else None
        })
        
    perf_data = {
        'total_wall_time': max_run_time,
        'algo_logic_time': 0.0 # Not tracked separately in QDFuzz
    }
        
    return merged_logs, perf_data

def deduplicate_log(merged_logs):
    """Deduplicate logs with tuple hashing, prioritizing crashes."""
    state_to_entry = {}
    
    for entry in merged_logs:
        state = entry['mutate_state']
        if state is None: continue
        
        # Array to tuple safely
        if isinstance(state, np.ndarray):
            state_key = tuple(state.flatten())
        else:
            state_key = tuple(state)
        
        if state_key not in state_to_entry:
            state_to_entry[state_key] = entry
        else:
            old_entry = state_to_entry[state_key]
            if entry.get('did_crash', False) and not old_entry.get('did_crash', False):
                state_to_entry[state_key] = entry

    return list(state_to_entry.values())

def plot_cumulative_crashes_from_csv(csv_file):
    """Strictly plot cumulative unique crashes from raw CSV."""
    MAX_H = 12.0
    VIEW_LIMIT_H = 12.5
    
    unique_crashes = []
    seen_inputs = set()
    
    df = pd.read_csv(csv_file)
    init_df = df[df['mutation_count'] == 0]
    if not init_df.empty:
        fuzz_start_time = init_df['discovery_time'].max()
    else:
        fuzz_start_time = 0.0
        
    df = df.sort_values(by='discovery_time')
    
    for _, row in df.iterrows():
        if int(row['mutation_count']) == 0:
            continue
            
        raw_time = float(row['discovery_time'])
        relative_time = raw_time - fuzz_start_time
        
        if relative_time > MAX_H * 3600:
            continue
            
        if bool(row['is_faulty']):
            inp_str = str(row['input'])
            if inp_str not in seen_inputs:
                seen_inputs.add(inp_str)
                unique_crashes.append(relative_time)
                
    times = np.array(unique_crashes)
    
    markers_x_h = np.arange(2, MAX_H + 0.1, 2)
    plt.figure(figsize=(10, 6))
    plot_color = '#1f77b4' 

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
    print(f"  Explored Unique States:     {explored_unique_states} / {THEORETICAL_STATE_SPACE} Grid Bins")
    print(f"  State Space Coverage:       {state_space_coverage:.6f}%")
    
    if perf_data:
        total_t = perf_data['total_wall_time']
        algo_t = perf_data['algo_logic_time']
        overhead_ratio = (algo_t / total_t) * 100 if total_t > 0 else 0
        print(f"  Fuzzer Overhead Ratio:      {overhead_ratio:.2f}% (Coverage Algo Time / Total Wall Time)\n")

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
    print(f"  Survival Steps (Depth) - Mean:   {np.mean(raw_survival_steps):.1f} steps")
    print(f"  Survival Steps (Depth) - Median: {np.median(raw_survival_steps):.1f} steps")
    print(f"  Survival Steps Range:            [{np.min(raw_survival_steps)}, {np.max(raw_survival_steps)}] steps\n")
    
    # Draw Cumulative unique crashes over time
    plot_cumulative_crashes_from_csv(CSV_FILE)

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
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(reduced_data)
            # Sample for silhouette score if data is large
            sample_sz = 5000 if len(reduced_data) > 5000 else None
            best_score = silhouette_score(reduced_data, labels, sample_size=sample_sz, random_state=42)
            
            for k in range(3, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(reduced_data)
                score = silhouette_score(reduced_data, labels, sample_size=sample_sz, random_state=42)
                if score >= best_score * 1.20:
                    best_score = score
                    best_k = k
                    
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
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
        if name == "Output" and raw_lengths is not None:
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

    generation_counts = Counter(crash_generations)
    generations = range(1, max_gen + 2)
    counts = [generation_counts.get(gen, 0) for gen in generations]

    plt.figure(figsize=(10, 6))
    plt.bar(generations, counts, color='#ff7f0e', alpha=0.8, edgecolor='black', zorder=3)
    plt.title(f'Histogram of Unique Crash Generations ({METHOD_NAME})')
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
    merged_logs, perf_data = load_and_merge_qdfuzz_data(CSV_FILE, OBS_FILE)
    
    if not merged_logs:
        return
        
    deduplicated_log = deduplicate_log(merged_logs)
    if not deduplicated_log: 
        print("Deduplicated log is empty.")
        return
    
    analyze_and_plot_comprehensive_metrics(merged_logs, deduplicated_log, perf_data)
    plot_generation_histogram(deduplicated_log)

    print(f"All analysis and plotting completed for [{METHOD_NAME}]. Check the generated PNG files.")

if __name__ == "__main__":
    main()
