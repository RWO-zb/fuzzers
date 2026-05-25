import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist
from collections import Counter

INPUT_CSV = 'summary.csv'
TRAJ_DIR = 'trajectories'

PLOT_3_FILE = 'crash_generation_histogram.png'
PLOT_4_FILE = 'unique_crashes_over_time.png'
PLOT_6_FILE = 'crash_distance_boxplot.png'

# Ego behavior features saved by get_enhanced_state_vector:
# x, y, forward_x, forward_y, velocity_x/y/z, acceleration_x/y/z, route command.
OUTPUT_BEHAVIOR_COLS = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5

# Parse CARLA state string into a numerical feature vector
def parse_input_features(input_str):
    """
    Parses CARLA state string into a numerical feature vector.
    """
    if pd.isna(input_str) or str(input_str) == "None": 
        return None
    try:
        parts = str(input_str).split('|')
        if len(parts) < 2: return None
        
        ego_part = parts[0].split(':')[1].strip('[]')
        ego_vals = [float(x) for x in ego_part.split(',') if x]
        
        npc_part = parts[1].split(':')[1]
        if not npc_part or npc_part == 'None':
            npc_feats = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            coords = [float(x) for x in npc_part.replace('(', '').replace(')', '').split(',') if x]
            if not coords: 
                npc_feats = [0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                xs, ys = coords[0::2], coords[1::2]
                npc_feats = [
                    float(len(xs)), np.mean(xs), np.mean(ys), 
                    np.std(xs) if len(xs)>1 else 0.0, np.std(ys) if len(ys)>1 else 0.0
                ]
        return np.array(ego_vals + npc_feats)
    except: 
        return None

# Load data from summary CSV and filter for relevant phases
def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None
    try:
        df = pd.read_csv(csv_path)
        if 'phase' in df.columns:
            df = df[df['phase'].isin(['Phase2', 'RT'])]
            
        original_log = []
        for _, row in df.iterrows():
            entry = {
                'task_id': row.get('task_id', ''),
                'is_crash': not (row.get('success') in [True, 'True', 1, '1']),
                'crash_time': float(row['global_time']) if pd.notna(row.get('global_time')) else 0.0,
                'mutate_state_str': row.get('current_input', 'None'),
                'parent_depth': int(float(row['generation'])) if pd.notna(row.get('generation')) else 0,
                'avg_speed': float(row['avg_speed']) if pd.notna(row.get('avg_speed')) else 0.0,
                'steer_std': float(row['steer_std']) if pd.notna(row.get('steer_std')) else 0.0,
                'final_dist': float(row['final_dist']) if pd.notna(row.get('final_dist')) else 0.0,
            }
            original_log.append(entry)
        return original_log
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Deduplicate logs based on unique input state strings
def deduplicate_log(original_log_data):
    state_to_entry = {}
    for entry in original_log_data:
        raw_input = entry['mutate_state_str']
        if pd.isna(raw_input) or str(raw_input) == "None": 
            continue
        unique_key = str(raw_input).strip()
        entry_copy = entry.copy()

        if unique_key not in state_to_entry:
            state_to_entry[unique_key] = entry_copy
        else:
            old_entry = state_to_entry[unique_key]
            if entry_copy['is_crash'] and not old_entry['is_crash']:
                state_to_entry[unique_key] = entry_copy
       
    return list(state_to_entry.values())

# Analyze and plot comprehensive fuzzing metrics including diversity and efficiency
def analyze_and_plot_comprehensive_metrics(original_log, deduplicated_log):
    print(f"\n{'='*85}")
    print(f"{'Academic-Grade Crash & Diversity Analysis (crash = not success)':^85}")
    print(f"{'='*85}")
    
    total_mutations = len(original_log)
    total_valid_crashes = sum(1 for e in original_log if e['is_crash'])
    hit_ratio = (total_valid_crashes / total_mutations * 100) if total_mutations > 0 else 0
    
    print("[1. Global Fuzzing Metrics]")
    print(f"  Total Mutations Executed:   {total_mutations}")
    print(f"  Valid Crash Mutations:      {total_valid_crashes}")
    print(f"  Hit Ratio (Valid Rate):     {hit_ratio:.2f}%  <-- % of mutations leading to a crash\n")
    
    inputs, times, depths = [], [], []
    crash_speeds, crash_steers, crash_distances = [], [], []
    valid_task_ids = []
    
    for entry in deduplicated_log:
        if entry['is_crash']:
            feat = parse_input_features(entry['mutate_state_str'])
            if feat is not None:
                inputs.append(feat)
                times.append(entry['crash_time'])
                depths.append(entry['parent_depth'])
                crash_speeds.append(entry['avg_speed'])
                crash_steers.append(entry['steer_std'])
                crash_distances.append(entry['final_dist'])
                valid_task_ids.append(entry['task_id'])
                
    unique_crash_count = len(inputs)
    if unique_crash_count < 2:
        print(f"Not enough unique crash data to calculate metrics (Found {unique_crash_count}, needs >= 2).")
        return
        
    inputs = np.array(inputs)
    times = np.array(times)
    depths = np.array(depths)
    
    times_hrs = np.sort(times / 3600.0)
    max_time_hrs = max([e['crash_time'] for e in original_log]) / 3600.0
    if max_time_hrs <= 0: max_time_hrs = times_hrs[-1] if len(times_hrs) > 0 else 1.0

    intervals_hrs = np.diff(np.insert(times_hrs, 0, 0.0))
    mean_interval = np.mean(intervals_hrs)      
    median_interval = np.median(intervals_hrs)  
    
    print("[2. Basic Crash Efficiency & Episode Depth]")
    print(f"  Total Unique Crashes Discovered: {unique_crash_count}")
    print(f"  Mean Interval per Crash:         {mean_interval:.4f} hours (~{mean_interval*3600:.1f} sec)")
    print(f"  Median Interval per Crash:       {median_interval:.4f} hours (~{median_interval*3600:.1f} sec)")
    print(f"  Speed at Crash - Mean:           {np.mean(crash_speeds):.2f} m/s")
    print(f"  Steering Instability - Mean:     {np.mean(crash_steers):.4f} rad")
    print(f"  Distance to Target - Mean:       {np.mean(crash_distances):.1f} m\n")
    
    cumulative_crashes = np.arange(1, len(times_hrs) + 1)
    plt.figure(figsize=(10, 6))
    plt.step(times_hrs, cumulative_crashes, where='post', color='#D62728', linewidth=2.5)
    plt.fill_between(times_hrs, cumulative_crashes, step='post', color='#D62728', alpha=0.1)
    plt.title('Cumulative Unique Crashes Discovered Over Time')
    plt.xlabel('Time Elapsed (hours)')
    plt.ylabel('Number of Unique Crashing Inputs')
    plt.xlim(left=0, right=max_time_hrs)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(PLOT_4_FILE, dpi=300)
    plt.close()

    # Helper to compute diversity metrics using PCA and KMeans clustering
    def compute_diversity_metrics(data_matrix, times_array, name, raw_lengths=None):
        n_samples = data_matrix.shape[0]
        if n_samples < 5:
            print(f"Not enough samples for clustering {name}.")
            return
            
        n_components = min(n_samples, data_matrix.shape[1], 10) 
        scaled_data = StandardScaler().fit_transform(data_matrix)
        pca = PCA(n_components=n_components, random_state=42)
        reduced_data = pca.fit_transform(scaled_data)
        
        best_k = 1
        best_score = -1
        max_k = min(15, n_samples - 1) 
        
        if max_k >= 2:
            best_k = 2
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(reduced_data)
            best_score = silhouette_score(reduced_data, labels)
            
            for k in range(3, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(reduced_data)
                score = silhouette_score(reduced_data, labels)
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
        avg_inter_dist = np.mean(pdist(centroids, metric='euclidean')) if best_k > 1 else 0.0
            
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

        if name == "Output" and raw_lengths is not None:
            cluster_steps = [raw_lengths[labels == k] for k in range(best_k)]
            plt.figure(figsize=(10, 6))
            plt.boxplot(cluster_steps, tick_labels=[f"C{k+1}\n(n={len(c)})" for k, c in enumerate(cluster_steps)])
            plt.title('Distance to Target per Crash Cluster')
            plt.ylabel('Distance to Target (m)')
            plt.tight_layout()
            plt.savefig(PLOT_6_FILE, dpi=300)
            plt.close()

    compute_diversity_metrics(inputs, times, "Input")

    outputs_padded = []
    times_matched = []
    crash_distances_matched = []
    
    for i, t_id in enumerate(valid_task_ids):
        npz_path = os.path.join(TRAJ_DIR, f"{t_id}.npz")
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path, allow_pickle=True)
                states_seq = data['states']
                if states_seq.ndim != 2 or states_seq.shape[1] <= max(OUTPUT_BEHAVIOR_COLS):
                    continue
                states_seq = states_seq[:, OUTPUT_BEHAVIOR_COLS]
                outputs_padded.append(states_seq)
                times_matched.append(times[i])
                crash_distances_matched.append(crash_distances[i])
            except Exception:
                pass
                
    if outputs_padded:
        max_len = max(len(t) for t in outputs_padded)
        final_outputs = []
        for t in outputs_padded:
            pad_len = max_len - len(t)
            padded = np.pad(t, ((0, pad_len), (0, 0)), mode='constant') if pad_len > 0 else t
            final_outputs.append(padded.flatten())
        
        outputs_matrix = np.array(final_outputs)
        times_matched = np.array(times_matched)
        crash_distances_matched = np.array(crash_distances_matched)
        compute_diversity_metrics(outputs_matrix, times_matched, "Output", raw_lengths=crash_distances_matched)
    print(f"{'='*85}\n")

# Plot the distribution of crash generations
def plot_generation_histogram(deduplicated_log):
    crash_generations = []
    for entry in deduplicated_log:
        if entry['is_crash']:
            crash_generations.append(entry['parent_depth'])
            
    if not crash_generations: return

    avg_gen = np.mean(crash_generations)
    median_gen = np.median(crash_generations)
    max_gen = np.max(crash_generations)
    
    print(f"[Evolutionary Depth Analysis]")
    print(f"  Average Crash Generation (Mean):   {avg_gen:.2f}")
    print(f"  Median Crash Generation (Median):  {median_gen:.2f}")
    print(f"  Deepest Crash Found at Generation: {max_gen}")

    generation_counts = Counter(crash_generations)
    generations = range(0, max_gen + 2)
    counts = [generation_counts.get(gen, 0) for gen in generations]

    plt.figure(figsize=(10, 6))
    plt.bar(generations, counts, color='#1F77B4', alpha=0.8, edgecolor='black', width=0.8)
    plt.title('Histogram of Unique Crash Generations')
    plt.xlabel('Mutation Generation')
    plt.ylabel('Number of Unique Crashing Inputs')
    step = max(1, (max_gen // 20))
    plt.xticks(np.arange(0, max_gen + 2, step=step))
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(PLOT_3_FILE, dpi=300)
    plt.close()

# Main execution flow for data processing and analysis
def main():
    global INPUT_CSV
    if len(sys.argv) > 1:
        INPUT_CSV = sys.argv[1]
        
    original_log_data = load_data(INPUT_CSV)
    
    if not original_log_data: 
        print("Failed to load log data or Phase2/RT data is empty.")
        return
        
    deduplicated_log = deduplicate_log(original_log_data)
    if not deduplicated_log: 
        print("Log data is empty or invalid after deduplication.")
        return
    
    analyze_and_plot_comprehensive_metrics(original_log_data, deduplicated_log)
    plot_generation_histogram(deduplicated_log)

    print("All analysis and plotting completed. Check the generated PNG files.")

if __name__ == "__main__":
    main()
