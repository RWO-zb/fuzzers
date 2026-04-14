# =============================================================================
# --- Imports & Dependencies ---
# =============================================================================
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from collections import Counter
import os

# =============================================================================
# --- Global Configuration & Constants ---
# =============================================================================
LOG_FILE = 'selection_log.pkl'
PERF_FILE = 'perf_meta.pkl'
PLOT_3_FILE = 'crash_generation_histogram.png'
PLOT_4_FILE = 'unique_crashes_over_time.png'       
PLOT_6_FILE = 'survival_steps_boxplot.png'

# The theoretical state space of BipedalWalker mutation (15 dims, 3 values each)
THEORETICAL_STATE_SPACE = 3**15 

# =============================================================================
# --- Data Loading Utilities ---
# Safely loads serialized pickle files.
# =============================================================================
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None

# =============================================================================
# --- Data Deduplication Module ---
# Removes duplicated input states from the log to ensure we only analyze 
# strictly unique crashes and unique explored states.
# =============================================================================
def deduplicate_log(original_log_data):
    seen_mutate_states = set()
    deduplicated_log = []
    dtype_to_use = None
    expected_size = 0
    int32_size = 15 * np.dtype(np.int32).itemsize
    int64_size = 15 * np.dtype(np.int64).itemsize

    for entry in original_log_data:
        state = entry.get('mutate_state')
        if state is None: continue
            
        try:
            state_bytes = state.tobytes()
        except AttributeError:
            continue
            
        if dtype_to_use is None:
            if len(state_bytes) == int32_size:
                dtype_to_use = np.int32
                expected_size = int32_size
            elif len(state_bytes) == int64_size:
                dtype_to_use = np.int64
                expected_size = int64_size
            else:
                continue 
        
        if len(state_bytes) != expected_size:
            continue
            
        if state_bytes not in seen_mutate_states:
            seen_mutate_states.add(state_bytes)
            entry_copy = entry.copy() 
            entry_copy['mutate_state'] = state_bytes
            deduplicated_log.append(entry_copy)

    if dtype_to_use is None:
        return None, None, 0

    return deduplicated_log, dtype_to_use, expected_size

# =============================================================================
# --- Core Analysis Module (Efficiency & Diversity) ---
# The primary engine computing overhead, coverage, depth, and clustering metrics.
# =============================================================================
def analyze_and_plot_comprehensive_metrics(original_log, deduplicated_log, perf_data):
    print(f"\n{'='*85}")
    print(f"{'Academic-Grade Crash & Diversity Analysis (Strictly did_crash == True)':^85}")
    print(f"{'='*85}")
    
    # --- 1. Global Fuzzing Metrics (Overhead, Hit Ratio, Coverage) ---
    total_mutations = len(original_log)
    total_valid_crashes = sum(1 for e in original_log if e.get('did_crash', False))
    hit_ratio = (total_valid_crashes / total_mutations * 100) if total_mutations > 0 else 0
    
    explored_unique_states = len(deduplicated_log)
    state_space_coverage = (explored_unique_states / THEORETICAL_STATE_SPACE) * 100
    
    # [修改逻辑] 修复 KeyError 并适配新旧版本的时间统计字段
    if perf_data:
        total_t = perf_data['total_wall_time']
        sim_t = perf_data['env_sim_time']
        
        # 优先读取拆分后的新指标，如果不存在则回退到旧键名
        if 'generation_time' in perf_data:
            gen_t = perf_data['generation_time']
            eval_t = perf_data['evaluation_time']
            other_t = perf_data['other_logic_time']
            # 总算法开销 = 总时间 - 环境仿真时间
            algo_t = total_t - sim_t
        else:
            algo_t = perf_data.get('algo_logic_time', total_t - sim_t)
            gen_t, eval_t, other_t = None, None, None
            
        overhead_ratio = (algo_t / total_t) * 100 if total_t > 0 else 0
    else:
        overhead_ratio = 0.0
    
    print("[1. Overhead, Hit Ratio & State Space Coverage]")
    print(f"  Total Mutations Executed:   {total_mutations}")
    print(f"  Valid Crash Mutations:      {total_valid_crashes}")
    print(f"  Hit Ratio (Valid Rate):     {hit_ratio:.2f}%  <-- % of mutations leading to a crash")
    print(f"  Explored Unique States:     {explored_unique_states} / {THEORETICAL_STATE_SPACE}")
    print(f"  State Space Coverage:       {state_space_coverage:.6f}%")
    if perf_data:
        print(f"  Fuzzer Overhead Ratio:      {overhead_ratio:.2f}% <-- % time spent in logic vs physics")
        # [新增] 如果存在细分时间，则打印出来
        if gen_t is not None:
            print(f"    - Generation Logic:       {gen_t:.2f}s (Selection & Mutation)")
            print(f"    - Evaluation Overhead:    {eval_t - sim_t:.2f}s (GMM Coverage Logic)")
            print(f"    - Other System Logic:     {other_t:.2f}s\n")
        else:
            print(f"    - Total Logic Time:       {algo_t:.2f}s\n")
    else:
        print(f"  Fuzzer Overhead Ratio:      N/A (perf_meta.pkl not found)\n")

    # --- 2. Data Extraction for Crash Analysis ---
    inputs = []
    outputs = []
    times = []
    depths = []
    raw_survival_steps = [] 
    
    for entry in deduplicated_log:
        if entry.get('did_crash', False) == True:
            state = entry.get('mutate_state')
            traj = entry.get('output_trajectory')
            depth = entry.get('parent_depth', 0)
            t = entry.get('elapsed_time', 0.0)
            
            survival_len = entry.get('survival_steps', len(traj) if traj is not None else 0)
            
            if state is not None and traj is not None:
                if isinstance(state, bytes):
                    state_arr = np.frombuffer(state, dtype=np.int32 if len(state)==60 else np.int64)
                else:
                    state_arr = np.array(state)
                    
                inputs.append(state_arr)
                outputs.append(traj)
                times.append(t)
                depths.append(depth + 1)
                raw_survival_steps.append(survival_len) 
                
    unique_crash_count = len(inputs)
    if unique_crash_count < 2:
        print(f"Not enough crash data to calculate metrics (Found {unique_crash_count}, needs >= 2).")
        return
        
    inputs = np.array(inputs)
    times = np.array(times)
    depths = np.array(depths)
    raw_survival_steps = np.array(raw_survival_steps)
    
    times_hrs = np.sort(times / 3600.0)
    # 获取整个 original_log 的最大时间
    max_time_hrs = max([e.get('elapsed_time', 0.0) for e in original_log] + [0.0]) / 3600.0
    
    # --- 3. Basic Efficiency & Survival Depth Analysis ---
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
    
    cumulative_crashes = np.arange(1, len(times_hrs) + 1)
    plt.figure(figsize=(12, 7))
    plt.step(times_hrs, cumulative_crashes, where='post', color='darkred', linewidth=2, label='Unique Crash Inputs')
    plt.fill_between(times_hrs, cumulative_crashes, step='post', color='darkred', alpha=0.1)
    plt.title('Cumulative Unique Crashes Discovered Over Time')
    plt.xlabel('Time Elapsed (hours)')
    plt.ylabel('Number of Unique Crashing Inputs')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0, right=max_time_hrs)
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(PLOT_4_FILE)

    # --- 4. Trajectory Padding for Sequence Clustering ---
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
            # 兼容新旧版本的 numpy trapezoid 函数
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

        # 5d. Boxplot Generation (Survival Steps)
        if name == "Output" and raw_lengths is not None:
            cluster_steps = [raw_lengths[labels == k] for k in range(best_k)]
            plt.figure(figsize=(10, 6))
            plt.boxplot(cluster_steps, tick_labels=[f"Cluster {k+1}\n(n={len(cluster_steps[k])})" for k in range(best_k)])
            plt.title('Crash Episode Length (Survival Steps) Distribution per Fault Type')
            plt.ylabel('Timesteps until Crash')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.savefig(PLOT_6_FILE)

    compute_diversity_metrics(inputs, times, "Input")
    compute_diversity_metrics(outputs_padded, times, "Output", raw_lengths=raw_survival_steps)
    print(f"{'='*85}\n")

# =============================================================================
# --- Supplementary Plotting Module ---
# Plots the distribution of generations in which unique crashes were found.
# =============================================================================
def plot_generation_histogram(deduplicated_log):
    crash_generations = []
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            parent_depth = entry.get('parent_depth')
            if parent_depth is not None:
                crash_generations.append(parent_depth + 1)
            
    if not crash_generations: return


    # === 新增：打印平均演化深度 ===
    avg_gen = np.mean(crash_generations)
    median_gen = np.median(crash_generations)
    max_gen = np.max(crash_generations)
    print(f"\n[Evolutionary Depth Analysis]")
    print(f"  Average Crash Generation (Mean):   {avg_gen:.2f}")
    print(f"  Median Crash Generation (Median):  {median_gen:.2f}")
    print(f"  Deepest Crash Found at Generation: {max_gen}")
    # ==============================
    generation_counts = Counter(crash_generations)
    generations = range(0, max_gen + 2)
    counts = [generation_counts.get(gen, 0) for gen in generations]

    plt.figure(figsize=(12, 7))
    plt.bar(generations, counts, color='red', alpha=0.7, zorder=3)
    plt.title('Histogram of Unique Crash Generations')
    plt.xlabel('Mutation Generation')
    plt.ylabel('Number of Unique Crashing Inputs')
    step = max(1, (max_gen // 20))
    plt.xticks(np.arange(0, max_gen + 2, step=step))
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.savefig(PLOT_3_FILE)
    plt.close()

# =============================================================================
# --- Main Execution Flow ---
# =============================================================================
def main():
    original_log_data = load_data(LOG_FILE)
    perf_data = load_data(PERF_FILE) 
    
    if not original_log_data: 
        print("Failed to load log data.")
        return
        
    # -------------------------------------------------------------------
    # [修改/新增] 仅保留 Fuzz 变异阶段的数据 (过滤掉 parent_depth == 0)
    # -------------------------------------------------------------------
    original_log_data = [entry for entry in original_log_data if entry.get('parent_depth', 0) > 0]
    
    if not original_log_data:
        print("No fuzz stage data (parent_depth > 0) found in the log.")
        return
    print(f"Filtered data to Fuzz Stage Only: {len(original_log_data)} mutations remaining.")
    # -------------------------------------------------------------------
        
    deduplicated_log, dtype, expected_size = deduplicate_log(original_log_data)
    if not deduplicated_log: 
        print("Log data is empty or invalid.")
        return
    
    analyze_and_plot_comprehensive_metrics(original_log_data, deduplicated_log, perf_data)
    plot_generation_histogram(deduplicated_log)

    print("All analysis and plotting completed. Check the generated PNG files.")

if __name__ == "__main__":
    main()