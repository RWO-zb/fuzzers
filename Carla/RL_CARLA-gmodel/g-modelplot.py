import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist
from collections import Counter
import os
import re

INPUT_CSV = 'summary.csv'
PLOT_4_FILE = 'unique_crashes_over_time.png'       
PLOT_6_FILE = 'survival_steps_boxplot.png'

# Ego behavior features saved by get_enhanced_state_vector:
# x, y, forward_x, forward_y, velocity_x/y/z, acceleration_x/y/z, route command.
OUTPUT_BEHAVIOR_COLS = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12]

def parse_input_features(input_str):
    """
    Parses CARLA state string into a numerical feature vector.
    Format example: "Ego:[x,y,yaw]|NPCs:(x1,y1)..."
    """
    if pd.isna(input_str) or str(input_str) == "None":
        return None
    
    try:
        parts = str(input_str).split('|')
        if len(parts) < 2: return None
        
        # 1. Ego Features
        ego_part = parts[0].split(':')[1].strip('[]')
        ego_vals = [float(x) for x in ego_part.split(',') if x]
        
        # 2. NPC Features
        npc_part = parts[1].split(':')[1]
        if not npc_part or npc_part == 'None':
            npc_feats = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            raw_nums = npc_part.replace('(', '').replace(')', '').split(',')
            coords = [float(x) for x in raw_nums if x]
            
            if not coords:
                npc_feats = [0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                xs = coords[0::2]
                ys = coords[1::2]
                npc_feats = [
                    float(len(xs)),      # Count
                    np.mean(xs),         # Mean X
                    np.mean(ys),         # Mean Y
                    np.std(xs) if len(xs) > 1 else 0.0, # Std X
                    np.std(ys) if len(ys) > 1 else 0.0  # Std Y
                ]
                
        return np.array(ego_vals + npc_feats)
    except:
        return None

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        # Filter for Fuzzing Phase
        if 'phase' in df.columns:
            df = df[df['phase'] == 'Phase2']
        
        if 'elapsed_time' in df.columns:
            df = df.sort_values(by='elapsed_time')
            
        original_log = []
        for _, row in df.iterrows():
            entry = row.to_dict()
            is_success = str(entry.get('success', 'False')).lower() == 'true'
            entry['is_crash'] = not is_success
            raw_input = entry.get('input_post')
            entry['features'] = parse_input_features(raw_input) if pd.notna(raw_input) else None
            
            # Map generation/step from task_id
            task_id = str(entry.get('task_id', ''))
            match = re.search(r'_(\d+)', task_id)
            if match:
                entry['parent_depth'] = int(match.group(1))
            else:
                entry['parent_depth'] = 0
                
            original_log.append(entry)
            
        return original_log
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def deduplicate_log(original_log_data):
    state_to_entry = {}

    for entry in original_log_data:
        raw_input = entry.get('input_post')
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

def analyze_and_plot_comprehensive_metrics(original_log, deduplicated_log):
    print(f"\n{'='*85}")
    print(f"{'Academic-Grade Crash & Diversity Analysis (Strictly did_crash == True)':^85}")
    print(f"{'='*85}")
    
    # --- 1. Global Fuzzing Metrics ---
    total_mutations = len(original_log)
    total_valid_crashes = sum(1 for e in original_log if e['is_crash'])
    hit_ratio = (total_valid_crashes / total_mutations * 100) if total_mutations > 0 else 0
    
    print("[1. Global Fuzzing Metrics]")
    print(f"  Total Mutations Executed:   {total_mutations}")
    print(f"  Valid Crash Mutations:      {total_valid_crashes}")
    print(f"  Hit Ratio (Valid Rate):     {hit_ratio:.2f}%  <-- % of mutations leading to a crash\n")

    # --- 2. Data Extraction for Crash Analysis ---
    inputs = []
    outputs = []
    output_times = []
    output_crash_distances = []
    times = []
    depths = []
    crash_distances = []
    crash_speeds = []
    crash_steers = []
    
    start_time = float(original_log[0].get('elapsed_time', 0.0)) if original_log else 0.0
    
    for entry in deduplicated_log:
        if entry['is_crash']:
            feats = entry.get('features')
            t = float(entry.get('elapsed_time', start_time)) - start_time
            depth = entry.get('parent_depth', 0)
            
            final_dist = float(entry.get('final_dist', 0.0))
            avg_speed = float(entry.get('avg_speed', 0.0))
            steer_std = float(entry.get('steer_std', 0.0))
            
            task_id = entry.get('task_id', '')
            traj = None
            
            # Load trajectory if available
            csv_dir = os.path.dirname(INPUT_CSV)
            traj_path = os.path.join(csv_dir if csv_dir else '.', 'trajectories', f"{task_id}.npz")
            if os.path.exists(traj_path):
                try:
                    data = np.load(traj_path, allow_pickle=True)
                    states_seq = data['states']
                    if states_seq.ndim == 2 and states_seq.shape[1] > max(OUTPUT_BEHAVIOR_COLS):
                        traj = states_seq[:, OUTPUT_BEHAVIOR_COLS]
                except Exception:
                    traj = None
            
            if feats is not None:
                inputs.append(feats)
                times.append(max(0.0, t))
                depths.append(int(float(depth)))
                crash_distances.append(final_dist)
                crash_speeds.append(avg_speed)
                crash_steers.append(steer_std)
                if traj is not None:
                    outputs.append(traj)
                    output_times.append(max(0.0, t))
                    output_crash_distances.append(final_dist)
                
    unique_crash_count = len(inputs)
    if unique_crash_count < 2:
        print(f"Not enough crash data to calculate metrics (Found {unique_crash_count}, needs >= 2).")
        return
        
    inputs = np.array(inputs)
    times = np.array(times)
    depths = np.array(depths)
    crash_distances = np.array(crash_distances)
    crash_speeds = np.array(crash_speeds)
    crash_steers = np.array(crash_steers)
    
    times_hrs = np.sort(times / 3600.0)
    
    max_elapsed = max([float(e.get('elapsed_time', start_time)) for e in original_log] + [start_time])
    max_time_hrs = (max_elapsed - start_time) / 3600.0
    
    # --- 3. Basic Efficiency & Survival Depth Analysis ---
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
    plt.figure(figsize=(12, 7))
    plt.step(times_hrs, cumulative_crashes, where='post', color='darkred', linewidth=2, label='Unique Crash Inputs')
    plt.fill_between(times_hrs, cumulative_crashes, step='post', color='darkred', alpha=0.1)
    plt.title('Cumulative Unique Crashes Discovered Over Time')
    plt.xlabel('Time Elapsed (hours)')
    plt.ylabel('Number of Unique Crashing Inputs')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0, right=max_time_hrs if max_time_hrs > 0 else max(times_hrs)+0.1)
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(PLOT_4_FILE)
    plt.close()

    # --- 4. Trajectory Padding for Sequence Clustering ---
    outputs_padded = None
    if outputs:
        max_len = max(len(t) for t in outputs)
        padded_outputs = []
        for t in outputs:
            pad_len = max_len - len(t)
            padded = np.pad(t, ((0, pad_len), (0, 0)), mode='constant') if pad_len > 0 else t
            padded_outputs.append(padded.flatten())
        outputs_padded = np.array(padded_outputs)
    
    # --- 5. Advanced Diversity Quality Metrics (PCA + KMeans) ---
    def compute_diversity_metrics(data_matrix, times_array, name, raw_lengths=None):
        
        # 5a. Dimensionality Reduction & Optimal K Selection
        n_samples = data_matrix.shape[0]
        if name == "Output" and n_samples < 5:
            print(f"Not enough samples for clustering {name}.")
            return
        n_components = min(n_samples, data_matrix.shape[1], 10) 
        pca = PCA(n_components=n_components, random_state=42)
        pca_input = StandardScaler().fit_transform(data_matrix) if name == "Output" else data_matrix
        reduced_data = pca.fit_transform(pca_input)
        
        best_k = 1
        best_score = -1
        max_k = min(15 if name == "Output" else 20, n_samples - 1) 
        
        if max_k >= 2:
            best_k = 2
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(reduced_data)
            best_score = silhouette_score(reduced_data, labels)
            
            # The 20% threshold mechanism to prevent noisy improvements
            for k in range(3, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(reduced_data)
                score = silhouette_score(reduced_data, labels)
                if score >= best_score * 1.20:
                    best_score = score
                    best_k = k
                    
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced_data)
        
        # 5b. Cluster Distance & Entropy Calculation
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
        
        # 5c. Time-To-Discovery (TTD) & AUC Calculation
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

        # 5d. Boxplot Generation (Distance to Target)
        if name == "Output" and raw_lengths is not None:
            cluster_steps = [raw_lengths[labels == k] for k in range(best_k)]
            plt.figure(figsize=(10, 6))
            plt.boxplot(cluster_steps, tick_labels=[f"Cluster {k+1}\n(n={len(cluster_steps[k])})" for k in range(best_k)])
            plt.title('Distance to Target at Crash Distribution per Fault Type')
            plt.ylabel('Distance to Target (m)')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.savefig('crash_distance_boxplot.png')
            plt.close()

    compute_diversity_metrics(inputs, times, "Input")
    if outputs_padded is not None:
        compute_diversity_metrics(outputs_padded, np.array(output_times), "Output", raw_lengths=np.array(output_crash_distances))
    print(f"{'='*85}\n")

def print_generation_stats(deduplicated_log):
    """Print crash step/generation statistics without plotting (not meaningful for diffusion models)."""
    crash_steps = []
    for entry in deduplicated_log:
        if entry.get('is_crash', False):
            parent_depth = entry.get('parent_depth')
            if parent_depth is not None:
                crash_steps.append(parent_depth + 1)
            
    if not crash_steps: return

    avg_step = np.mean(crash_steps)
    median_step = np.median(crash_steps)
    max_step = np.max(crash_steps)
    
    print(f"\n[Crash Step Distribution (G-Model has no generational structure)]")
    print(f"  Average Crash Step (Mean):   {avg_step:.2f}")
    print(f"  Median Crash Step (Median):  {median_step:.2f}")
    print(f"  Deepest Crash at Step:       {max_step}")

def main():
    import sys
    global INPUT_CSV
    if len(sys.argv) > 1:
        INPUT_CSV = sys.argv[1]
        
    original_log_data = load_data(INPUT_CSV)
    
    if not original_log_data: 
        print("Failed to load log data or Phase2 data is empty.")
        return
        
    deduplicated_log = deduplicate_log(original_log_data)
    if not deduplicated_log: 
        print("Log data is empty or invalid after deduplication.")
        return
    
    analyze_and_plot_comprehensive_metrics(original_log_data, deduplicated_log)
    print_generation_stats(deduplicated_log)

    print("All analysis and plotting completed. Check the generated PNG files.")

if __name__ == "__main__":
    main()
