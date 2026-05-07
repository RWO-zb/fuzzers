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
import os
from collections import Counter

# =============================================================================
# --- Global Configuration & Constants ---
# =============================================================================
LOG_FILE = 'selection_log.pkl'
OBS_FILE = 'obs_sequences.pkl'
PLOT_CUMULATIVE_FILE = 'MountainCar_Unique_Crashes_Time.png'       
PLOT_GEN_FILE = 'MountainCar_Crash_Generations.png'
PLOT_SURVIVAL_FILE = 'MountainCar_Survival_Steps.png'

# 采用 50x50 的网格划分作为理论状态空间大小 (借鉴 RQ2 逻辑)
THEORETICAL_STATE_SPACE = 50 * 50 

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
# --- Data Merging & Deduplication Module ---
# =============================================================================
def merge_and_deduplicate(logs, obs_seqs):
    """
    将 selection_log 和 obs_sequences 合并，并进行浮点数级别的去重。
    """
    if len(logs) != len(obs_seqs):
        print(f"Warning: Logs count ({len(logs)}) and Obs count ({len(obs_seqs)}) mismatch. Truncating to min.")
    
    min_len = min(len(logs), len(obs_seqs))
    state_to_entry = {}
    
    for i in range(min_len):
        entry = logs[i]
        traj = obs_seqs[i]
        state = entry.get('mutate_state')
        
        if state is None: continue
            
        # 对于 2D 浮点数，保留 4 位小数作为哈希键，避免浮点数精度漂移导致假性重复
        state_key = (round(state[0], 4), round(state[1], 4))
        
        entry_copy = entry.copy()
        entry_copy['output_trajectory'] = traj
        entry_copy['survival_steps'] = len(traj)
        
        # 去重与优先级覆盖：如果同一个输入状态下，新的记录是 Crash，则覆盖旧的安全记录
        if state_key not in state_to_entry:
            state_to_entry[state_key] = entry_copy
        else:
            old_entry = state_to_entry[state_key]
            if entry_copy.get('did_crash', False) and not old_entry.get('did_crash', False):
                state_to_entry[state_key] = entry_copy

    return list(state_to_entry.values())

# =============================================================================
# --- Core Diversity & Efficiency Analysis ---
# =============================================================================
def compute_diversity_metrics(data_matrix, times_array, name, raw_lengths=None):
    n_samples = data_matrix.shape[0]
    if n_samples < 2:
        print(f"[Warning] Not enough data for {name} diversity analysis (n={n_samples})")
        return 0, 0.0, 0.0

    # PCA 降维 (MountainCar 特征少，保留维度相应降低)
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
        
        # 轮廓系数必须提升 20% 才增加聚类数量，避免噪声过拟合
        for k in range(3, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(reduced_data)
            score = silhouette_score(reduced_data, labels)
            if score >= best_score * 1.20:
                best_score = score
                best_k = k
                
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reduced_data)
    
    # 熵计算 (信息熵)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    
    print(f"[Metrics - {name} Diversity]")
    print(f"  Clusters Discovered (Unique Modes): {best_k}")
    print(f"  Silhouette Score (Quality):         {best_score:.4f}")
    print(f"  Entropy (Distribution Evenness):    {entropy:.4f}\n")

    # 绘制生存步数箱线图 (针对 Output Trajectory)
    if name == "Output" and raw_lengths is not None:
        cluster_steps = [raw_lengths[labels == k] for k in range(best_k)]
        plt.figure(figsize=(10, 6))
        plt.boxplot(cluster_steps, tick_labels=[f"C{k+1}\n(n={len(cluster_steps[k])})" for k in range(best_k)])
        plt.title('MountainCar: Survival Steps Distribution per Failure Type')
        plt.ylabel('Timesteps until Termination')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.savefig(PLOT_SURVIVAL_FILE)
        plt.close()

    return best_k, entropy, best_score

def analyze_and_plot(dedup_log):
    print(f"\n{'='*70}")
    print(f"{'MountainCar: INDAGO-Nexus Evaluation Metrics':^70}")
    print(f"{'='*70}")
    
    total_mutations = len(dedup_log)
    crashes = [e for e in dedup_log if e.get('did_crash', False)]
    total_valid_crashes = len(crashes)
    hit_ratio = (total_valid_crashes / total_mutations * 100) if total_mutations > 0 else 0
    state_space_coverage = (total_mutations / THEORETICAL_STATE_SPACE) * 100
    
    print("[Global Execution Metrics]")
    print(f"  Explored Unique States: {total_mutations} / {THEORETICAL_STATE_SPACE} Grid Bins")
    print(f"  State Space Coverage:   {state_space_coverage:.2f}%")
    print(f"  Valid Crashes Found:    {total_valid_crashes}")
    print(f"  Hit Ratio:              {hit_ratio:.2f}%\n")
    
    if total_valid_crashes < 2:
        print("Not enough crashes to run PCA/KMeans diversity analysis.")
        return

    # 数据提取
    inputs, outputs, times, depths, survival_steps = [], [], [], [], []
    for c in crashes:
        inputs.append(c['mutate_state'])
        outputs.append(np.array(c['output_trajectory']).flatten())
        depths.append(c.get('parent_depth', 0) + 1)
        survival_steps.append(c['survival_steps'])
        # 处理时间：如果没有记录则默认放在末尾
        t = c.get('crash_time')
        times.append(t if t is not None else 0.0)

    inputs = np.array(inputs)
    times = np.array(times)
    depths = np.array(depths)
    survival_steps = np.array(survival_steps)

    # 轨迹长度对齐 (Zero Padding)
    max_len = max(len(t) for t in outputs)
    padded_outputs = [np.pad(t, (0, max_len - len(t)), mode='constant') for t in outputs]
    outputs_padded = np.array(padded_outputs)

    # 1. 效率与深度指标
    avg_gen = np.mean(depths)
    valid_times = times[times > 0]
    ttf = np.min(valid_times) if len(valid_times) > 0 else 0.0
    
    print("[Efficiency & Depth Metrics]")
    print(f"  Time-To-Failure (TTF):      {ttf:.2f} sec")
    print(f"  Average Evolutionary Depth: {avg_gen:.2f} Generations\n")

    # 2. 多样性指标计算
    compute_diversity_metrics(inputs, times, "Input")
    compute_diversity_metrics(outputs_padded, times, "Output", raw_lengths=survival_steps)

    # 3. 绘图: 随时间累积的独特 Crash 数量
    if len(valid_times) > 0:
        times_hrs = np.sort(valid_times / 3600.0)
        cumulative_crashes = np.arange(1, len(times_hrs) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.step(times_hrs, cumulative_crashes, where='post', color='#d62728', linewidth=2, label='CureFuzz')
        plt.fill_between(times_hrs, cumulative_crashes, step='post', color='#d62728', alpha=0.1)
        plt.title('MountainCar: Cumulative Unique Crashes Over Time')
        plt.xlabel('Time Elapsed (hours)')
        plt.ylabel('Number of Unique Crashes')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_CUMULATIVE_FILE)
        plt.close()

    # 4. 绘图: Crash 所属的变异代数直方图
    generation_counts = Counter(depths)
    if generation_counts:
        max_gen = max(generation_counts.keys())
        generations = range(1, max_gen + 2)
        counts = [generation_counts.get(gen, 0) for gen in generations]

        plt.figure(figsize=(10, 6))
        plt.bar(generations, counts, color='#1f77b4', alpha=0.8, edgecolor='black')
        plt.title('MountainCar: Distribution of Crash Evolutionary Depths')
        plt.xlabel('Mutation Generation')
        plt.ylabel('Number of Crashing Inputs')
        plt.xticks(np.arange(1, max_gen + 2, step=max(1, max_gen//10)))
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(PLOT_GEN_FILE)
        plt.close()

def main():
    print("Loading data...")
    logs = load_data(LOG_FILE)
    obs_seqs = load_data(OBS_FILE)
    
    if not logs or not obs_seqs:
        print("Data loading failed. Ensure 'selection_log.pkl' and 'obs_sequences.pkl' exist.")
        return
        
    dedup_log = merge_and_deduplicate(logs, obs_seqs)
    analyze_and_plot(dedup_log)
    print("Analysis complete. Check the generated .png files.")

if __name__ == "__main__":
    main()