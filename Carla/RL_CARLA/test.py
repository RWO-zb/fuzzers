import os
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt

INPUT_CSV = 'summary.csv'
TRAJ_DIR = 'trajectories'

# =============================================================================
# --- 数据解析工具 ---
# =============================================================================
def parse_input_features(input_str):
    """从 input_post 提取特征向量，与你原来的 plot.py 逻辑一致"""
    if pd.isna(input_str) or str(input_str) == "None": return None
    try:
        parts = str(input_str).split('|')
        if len(parts) < 2: return None
        ego_vals = [float(x) for x in parts[0].split(':')[1].strip('[]').split(',') if x]
        npc_part = parts[1].split(':')[1]
        if not npc_part or npc_part == 'None':
            npc_feats = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            coords = [float(x) for x in npc_part.replace('(', '').replace(')', '').split(',') if x]
            if not coords: npc_feats = [0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                xs, ys = coords[0::2], coords[1::2]
                npc_feats = [float(len(xs)), np.mean(xs), np.mean(ys), np.std(xs) if len(xs)>1 else 0.0, np.std(ys) if len(ys)>1 else 0.0]
        return np.array(ego_vals + npc_feats)
    except: return None

# =============================================================================
# --- 核心指标计算引擎 (完全移植自 test1.py) ---
# =============================================================================
def compute_diversity_metrics(data_matrix, name):
    print(f"\n[{name} Diversity Analysis]")
    n_samples = data_matrix.shape[0]
    
    if n_samples < 5:
        print("  Not enough crash samples for clustering (needs >= 5).")
        return

    # 1. PCA 降维 (防止维度灾难，特别是轨迹数据)
    n_components = min(n_samples, data_matrix.shape[1], 10) 
    pca = PCA(n_components=n_components, random_state=42)
    reduced_data = pca.fit_transform(data_matrix)
    
    # 2. KMeans 寻找最佳 K 值 (基于 Silhouette Score 提升 20% 阈值)
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
    
    # 3. 计算簇内方差 (Intra) 和 簇间距离 (Inter) 和 熵 (Entropy)
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
    
    print(f"  Total Valid Crashes Analyzed:      {n_samples}")
    print(f"  Clusters Discovered (K*):          {best_k}")
    print(f"  Absolute Silhouette Score:         {best_score:.4f} (Range [-1, 1])")
    print(f"  Avg Intra-Cluster Dist (Variance): {avg_intra_dist:.4f} (Larger = diverse states within same fault type)")
    print(f"  Avg Inter-Cluster Dist (Distance): {avg_inter_dist:.4f} (Larger = clearer distinction between fault types)")
    print(f"  Entropy (Distribution Evenness):   {entropy:.4f}")
    
    return labels

# =============================================================================
# --- 主执行流程 ---
# =============================================================================
def main():
    print(f"{'='*80}\n{'CARLA Crash Diversity Assessment (PCA + KMeans)':^80}\n{'='*80}")
    
    # 1. 加载并过滤数据
    df = pd.read_csv(INPUT_CSV)
    df = df[df['success'] == False] # 只分析发生故障(Crash/Timeout)的场景
    print(f"Found {len(df)} total failures in summary.csv.")

    # ---------------------------------------------------------
    # 分析 1: 输入状态多样性 (Input Diversity)
    # ---------------------------------------------------------
    input_features = []
    valid_task_ids = []
    seen_inputs = set()
    
    for _, row in df.iterrows():
        raw_input = row.get('input_post')
        if pd.isna(raw_input) or str(raw_input) == "None": continue
        
        unique_key = str(raw_input).strip()
        if unique_key not in seen_inputs:
            seen_inputs.add(unique_key)
            feat = parse_input_features(unique_key)
            if feat is not None:
                input_features.append(feat)
                valid_task_ids.append(row['task_id'])
                
    if input_features:
        compute_diversity_metrics(np.array(input_features), "Input (Scene Generation)")

    # ---------------------------------------------------------
    # 分析 2: 输出轨迹行为多样性 (Output Trajectory Diversity)
    # ---------------------------------------------------------
    outputs_padded = []
    raw_survival_steps = []
    
    crash_trajs = []
    for t_id in valid_task_ids:
        npz_path = os.path.join(TRAJ_DIR, f"{t_id}.npz")
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                states_seq = data['states'] # 获取序列化的状态数组
                crash_trajs.append(states_seq)
                raw_survival_steps.append(len(states_seq))
            except Exception as e:
                pass
                
    if crash_trajs:
        # 对齐所有轨迹（Padding）
        max_len = max(len(t) for t in crash_trajs)
        for t in crash_trajs:
            pad_len = max_len - len(t)
            # 以 0 填充以对齐长度
            padded = np.pad(t, ((0, pad_len), (0, 0)), mode='constant') if pad_len > 0 else t
            outputs_padded.append(padded.flatten()) # 拍平成一维向量用于 PCA
            
        outputs_matrix = np.array(outputs_padded)
        labels = compute_diversity_metrics(outputs_matrix, "Output (Crash Trajectories)")
        
        # 可选：绘制存活步数分布箱线图 (同 test1.py)
        if labels is not None:
            best_k = len(np.unique(labels))
            raw_survival_steps = np.array(raw_survival_steps)
            cluster_steps = [raw_survival_steps[labels == k] for k in range(best_k)]
            plt.figure(figsize=(10, 6))
            plt.boxplot(cluster_steps, tick_labels=[f"Cluster {k+1}\n(n={len(c)})" for k, c in enumerate(cluster_steps)])
            plt.title('Survival Steps Distribution per Crash Cluster')
            plt.ylabel('Timesteps until Crash/Stop')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.savefig('carla_survival_steps_boxplot.png')
            print("\n  [+] Saved 'carla_survival_steps_boxplot.png'")

    print(f"{'='*80}")

if __name__ == "__main__":
    main()