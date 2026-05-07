import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from collections import Counter
import matplotlib.pyplot as plt

# =============================================================================
# --- 配置与常量 ---
# =============================================================================
INPUT_CSV = 'summary.csv'
TRAJ_DIR = 'trajectories'

PLOT_CUMULATIVE_FILE = 'carla_cumulative_unique_crashes.png'
PLOT_GEN_FILE = 'carla_crash_generation_hist.png'
PLOT_SURVIVAL_FILE = 'carla_survival_steps.png'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5

# =============================================================================
# --- 数据解析工具 ---
# =============================================================================
def parse_input_features(input_str):
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

# =============================================================================
# --- 核心指标计算引擎 (包含 TTD 与 AUC) ---
# =============================================================================
def compute_diversity_metrics(data_matrix, times_array, max_time_hrs, name, raw_lengths=None):
    print(f"\n[{name} Diversity & Efficiency Analysis]")
    n_samples = data_matrix.shape[0]
    
    if n_samples < 5:
        print(f"  Not enough crash samples for clustering (needs >= 5). Current: {n_samples}")
        return None, 0

    # 1. PCA 降维
    n_components = min(n_samples, data_matrix.shape[1], 10) 
    pca = PCA(n_components=n_components, random_state=42)
    reduced_data = pca.fit_transform(data_matrix)
    
    # 2. KMeans 寻找最佳 K 值
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
    
    # 3. 簇内方差、簇间距离、信息熵
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

    # 4. 恢复 test1.py 中的 Step 5c: TTD & AUC Calculation
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
    
    print(f"  Total Valid Crashes Analyzed:        {n_samples}")
    print(f"  Clusters Discovered (Unique Bugs):   {best_k}")
    print(f"  Absolute Silhouette Score:           {best_score:.4f} (Range [-1, 1])")
    print(f"  Avg Intra-Cluster Dist (Variance):   {avg_intra_dist:.4f}")
    print(f"  Avg Inter-Cluster Dist (Distance):   {avg_inter_dist:.4f}")
    print(f"  Entropy (Distribution Evenness):     {entropy:.4f}")
    # 补回的关键输出
    print(f"  Mean Time-to-Discovery per Category: {mean_ttd:.4f} hours")
    print(f"  Diversity AUC (Clusters vs Time):    {auc_val:.4f} (category*hours)")

    if name == "Output (Crash Trajectories)" and raw_lengths is not None:
        cluster_steps = [raw_lengths[labels == k] for k in range(best_k)]
        plt.figure(figsize=(10, 6))
        plt.boxplot(cluster_steps, tick_labels=[f"C{k+1}\n(n={len(c)})" for k, c in enumerate(cluster_steps)])
        plt.title('CARLA: Survival Steps Distribution per Failure Type')
        plt.ylabel('Timesteps until Collision / Timeout')
        plt.tight_layout()
        plt.savefig(PLOT_SURVIVAL_FILE, dpi=300)
        plt.close()

    return labels, best_k

# =============================================================================
# --- 主执行流程 ---
# =============================================================================
def main():
    print(f"{'='*80}\n{'CARLA INDAGO-Nexus Strict Diversity Assessment':^80}\n{'='*80}")
    
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] 找不到文件: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    if 'phase' in df.columns:
        df = df[df['phase'] == 'Phase2']
    
    df_crash = df[df['success'] == False].copy()
    
    if 'elapsed_time' in df_crash.columns:
        df_crash = df_crash.sort_values(by='elapsed_time')

    print(f"Found {len(df_crash)} total Fuzzing failures in {INPUT_CSV}.")

    input_features = []
    valid_task_ids = []
    times = []
    depths = []
    seen_inputs = set()
    
    for _, row in df_crash.iterrows():
        raw_input = row.get('input_post')
        if pd.isna(raw_input) or str(raw_input) == "None": 
            continue
        
        unique_key = str(raw_input).strip()
        if unique_key not in seen_inputs:
            seen_inputs.add(unique_key)
            feat = parse_input_features(unique_key)
            if feat is not None:
                input_features.append(feat)
                t_id = str(row['task_id']).replace('.npz', '').strip()
                valid_task_ids.append(t_id)
                
                t = row.get('elapsed_time')
                times.append(float(t) if pd.notna(t) else 0.0)
                
                gen = row.get('mutation_generation')
                depths.append(int(float(gen)) if pd.notna(gen) else 0)

    inputs_matrix = np.array(input_features)
    times_array = np.array(times)
    
    # 获取最大运行时间 (转换为小时)，用于计算 AUC
    max_time_hrs = np.max(times_array) / 3600.0 if len(times_array) > 0 else 0.0

    if len(inputs_matrix) > 0:
        labels_input, unique_crash_count = compute_diversity_metrics(inputs_matrix, times_array, max_time_hrs, "Input (Scene Generation)")
    else:
        print("No valid input features found.")
        return

    if unique_crash_count > 0 and len(times_array) > 0:
        valid_times = times_array[times_array > 0]
        if len(valid_times) > 0:
            times_hrs = np.sort(valid_times / 3600.0)
            cumulative_crashes = np.arange(1, len(times_hrs) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.step(times_hrs, cumulative_crashes, where='post', color='#D62728', linewidth=2.5)
            plt.fill_between(times_hrs, cumulative_crashes, step='post', color='#D62728', alpha=0.1)
            plt.title('CARLA: Cumulative Unique Crashes Over Time')
            plt.xlabel('Time Elapsed (hours)')
            plt.ylabel('Number of Unique Crashing Inputs')
            plt.xlim(left=0, right=max_time_hrs)
            plt.ylim(bottom=0)
            plt.tight_layout()
            plt.savefig(PLOT_CUMULATIVE_FILE, dpi=300)
            plt.close()

    if depths:
        generation_counts = Counter(depths)
        max_gen = max(generation_counts.keys())
        generations = range(0, max_gen + 1)
        counts = [generation_counts.get(gen, 0) for gen in generations]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(generations, counts, color='#1F77B4', alpha=0.8, edgecolor='black', width=0.8)
        plt.title('CARLA: Histogram of Failure Mutation Generations')
        plt.xlabel('Mutation Generation Depth')
        plt.ylabel('Number of Unique Failures')
        
        step = max(1, max_gen // 15)
        plt.xticks(np.arange(0, max_gen + 1, step))
        plt.tight_layout()
        plt.savefig(PLOT_GEN_FILE, dpi=300)
        plt.close()

    outputs_padded = []
    raw_survival_steps = []
    crash_trajs = []
    
    for t_id in valid_task_ids:
        npz_path = os.path.join(TRAJ_DIR, f"{t_id}.npz")
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path, allow_pickle=True)
                states_seq = data['states'] 
                crash_trajs.append(states_seq)
                raw_survival_steps.append(len(states_seq))
            except Exception:
                pass
                
    if crash_trajs:
        max_len = max(len(t) for t in crash_trajs)
        for t in crash_trajs:
            pad_len = max_len - len(t)
            padded = np.pad(t, ((0, pad_len), (0, 0)), mode='constant') if pad_len > 0 else t
            outputs_padded.append(padded.flatten()) 
            
        outputs_matrix = np.array(outputs_padded)
        raw_survival_steps = np.array(raw_survival_steps)
        
        compute_diversity_metrics(outputs_matrix, times_array, max_time_hrs, "Output (Crash Trajectories)", raw_lengths=raw_survival_steps)

if __name__ == "__main__":
    main()