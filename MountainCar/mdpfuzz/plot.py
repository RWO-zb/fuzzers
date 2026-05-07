# =============================================================================
# --- Imports & Dependencies ---
# =============================================================================
import json
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
# 【配置区】请修改这里的 BASE_PREFIX 指向你想分析的实验结果前缀 (无需带 _logs.txt / _obs.txt 后缀)
# 示例 1 (MDPFuzz): 'logs/MC_DQN_NoCov_5_0.01_0.1_0_12h'
# 示例 2 (RT):      'logs/MC_DQN_RT_1022_10000it'
BASE_PREFIX = 'MC_DQN_NoCov_5_0.01_0.1_0_12h'

# 自动推断当前是 MDPFuzz 还是 Random Testing
IS_RT = '_RT_' in BASE_PREFIX
METHOD_NAME = 'Random Testing' if IS_RT else 'MDPFuzz'

LOG_FILE = f"{BASE_PREFIX}_logs.txt"
OBS_FILE = f"{BASE_PREFIX}_obs.txt"

PLOT_CUMULATIVE_FILE = f"{METHOD_NAME.replace(' ', '_')}_unique_crashes_over_time.png"       
PLOT_GEN_FILE = f"{METHOD_NAME.replace(' ', '_')}_crash_generation_histogram.png"
PLOT_SURVIVAL_FILE = f"{METHOD_NAME.replace(' ', '_')}_survival_steps_boxplot.png"

# MountainCar 理论状态空间划分 (50x50 网格)
THEORETICAL_STATE_SPACE = 50 * 50 

# =============================================================================
# --- Log Parser & Merging Module ---
# =============================================================================
def load_and_merge_mdpfuzz_data(log_file, obs_file):
    """
    解析并合并 _logs.txt 和 _obs.txt，兼容 RT 中字段为 'None' 的情况。
    """
    if not os.path.exists(log_file) or not os.path.exists(obs_file):
        print(f"Error: Log or Obs file not found for prefix '{BASE_PREFIX}'.")
        return None, None
        
    print(f"Parsing log files for [{METHOD_NAME}]...")
    
    # 解析 _logs.txt
    logs = []
    with open(log_file, 'r') as f:
        headers = f.readline().strip().split('; ')
        for line in f:
            line = line.strip()
            if not line: continue
            vals = line.split('; ')
            if len(vals) < len(headers): continue
            logs.append(dict(zip(headers, vals)))
            
    # 解析 _obs.txt
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
            
    if len(logs) != len(obs_data):
        print(f"Warning: Logs count ({len(logs)}) and Obs count ({len(obs_data)}) mismatch. Truncating to min.")
        
    merged_logs = []
    min_len = min(len(logs), len(obs_data))
    
    total_algo_time = 0.0
    max_run_time = 0.0
    
    for i in range(min_len):
        log_row = logs[i]
        obs_info, traj = obs_data[i]
        
        mutate_state = np.array(obs_info['Input'])
        did_crash = obs_info.get('Oracle', False)
        parent_depth = obs_info.get('Generation', 0)
        survival_steps = obs_info.get('Steps', len(traj))
        
        # 提取时间数据，处理 RT 中可能出现的 'None'
        run_time = log_row.get('RunTime', 'None')
        run_time = float(run_time) if run_time != 'None' else 0.0
        max_run_time = max(max_run_time, run_time)
        
        crash_time = log_row.get('CrashTime', 'None')
        crash_time = float(crash_time) if crash_time != 'None' else run_time
        
        algo_time = log_row.get('CoverageTime', 'None')
        algo_time = float(algo_time) if algo_time != 'None' else 0.0
        total_algo_time += algo_time
        
        merged_logs.append({
            'mutate_state': mutate_state,
            'did_crash': did_crash,
            'parent_depth': parent_depth,
            'survival_steps': survival_steps,
            'output_trajectory': np.array(traj) if did_crash else None,
            'run_time': run_time,
            'crash_time': crash_time
        })
        
    perf_data = {
        'total_wall_time': max_run_time,
        'algo_logic_time': total_algo_time
    }
        
    return merged_logs, perf_data

# =============================================================================
# --- Data Deduplication Module ---
# =============================================================================
def deduplicate_log(merged_logs):
    """严格的元组哈希去重与优先级覆盖 (Crash 覆盖 Safe)"""
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

# =============================================================================
# --- Core Analysis Module (Efficiency & Diversity) ---
# =============================================================================
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
    
    if IS_RT:
        print(f"  Fuzzer Overhead Ratio:      N/A (Random Testing has no algorithmic overhead)\n")
    elif perf_data:
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
    
    # 绘图: 随时间累积的独特 Crash 数量
    cumulative_crashes = np.arange(1, len(times_hrs) + 1)
    plt.figure(figsize=(10, 6))
    plot_color = '#1f77b4' if IS_RT else '#ff7f0e'
    plt.step(times_hrs, cumulative_crashes, where='post', color=plot_color, linewidth=2, label=f'{METHOD_NAME} Unique Crashes')
    plt.fill_between(times_hrs, cumulative_crashes, step='post', color=plot_color, alpha=0.1)
    plt.title(f'Cumulative Unique Crashes Discovered Over Time ({METHOD_NAME})')
    plt.xlabel('Time Elapsed (hours)')
    plt.ylabel('Number of Unique Crashing Inputs')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0, right=max_time_hrs)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_CUMULATIVE_FILE)
    plt.close()

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
            # 加速: 数据量过大时抽样计算轮廓系数
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

        # 仅针对 Output 绘制存活步数箱线图
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

# =============================================================================
# --- Evolutionary Depth Module ---
# =============================================================================
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

# =============================================================================
# --- Main Execution Flow ---
# =============================================================================
def main():
    print(f"Loading files for prefix: {BASE_PREFIX}")
    merged_logs, perf_data = load_and_merge_mdpfuzz_data(LOG_FILE, OBS_FILE)
    
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