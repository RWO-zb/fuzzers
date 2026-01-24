import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import os

# --- 配置常量 ---
LOG_FILE = 'selection_log.pkl'
PLOT_1_FILE = 'crashes_over_unique_inputs.png'
PLOT_2_FILE = 'full_input_space_tsne.png'
PLOT_3_FILE = 'crash_generation_histogram.png'
PLOT_4_FILE = 'crashes_over_time.png'
PLOT_5_FILE = 'behaviour_coverage_heatmap.png' 

def load_data(file_path):
    """加载 pickle 日志文件"""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None

def deduplicate_log(original_log_data):
    """
    对日志进行去重，保留 'mutate_state' 对应的唯一条目。
    同时保留条目中的所有其他字段（包括新增的 bd_distance 等）。
    """
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
            
        # 自动推断数据类型
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
            # 复制条目以防止修改原始数据，并确保 mutate_state 格式统一
            entry_copy = entry.copy() 
            entry_copy['mutate_state'] = state_bytes
            deduplicated_log.append(entry_copy)

    if dtype_to_use is None:
        return None, None, 0

    return deduplicated_log, dtype_to_use, expected_size

def print_data_statistics(original_log, deduplicated_log):
    """
    [新增] 计算并打印去重前后的统计数据
    """
    # 1. 计算原始统计 (Before Deduplication)
    original_count = len(original_log)
    original_crashes = sum(1 for entry in original_log if entry.get('did_crash', False))
    
    # 2. 计算去重后统计 (After Deduplication)
    unique_count = len(deduplicated_log)
    unique_crashes = sum(1 for entry in deduplicated_log if entry.get('did_crash', False))
    
    # 3. 打印对比表格
    print(f"\n{'='*50}")
    print(f"{'Data Statistics Report':^50}")
    print(f"{'='*50}")
    
    # Header
    print(f"{'Metric':<25} | {'Original (Raw)':<10} | {'Deduplicated (Unique)':<10}")
    print(f"{'-'*25} | {'-'*14} | {'-'*20}")
    
    # Test Cases Row
    print(f"{'Total Test Cases':<25} | {original_count:<14} | {unique_count:<20}")
    
    # Crashes Row
    print(f"{'Total Crashes':<25} | {original_crashes:<14} | {unique_crashes:<20}")
    
    print(f"{'-'*53}")
    
    # Additional Stats
    if original_count > 0:
        duplication_rate = (1 - unique_count / original_count) * 100
        print(f" Duplication Rate: {duplication_rate:.2f}%")
    
    if original_crashes > 0:
        crash_duplication_rate = (1 - unique_crashes / original_crashes) * 100
        print(f" Crash Redundancy: {crash_duplication_rate:.2f}% (Redundant crashes removed)")
        
    print(f"{'='*50}\n")

def plot_crash_trend(deduplicated_log):
    """[图表 1] 绘制故障发现趋势"""
    cumulative_crashes_list = []
    current_crash_count = 0
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            current_crash_count += 1
        cumulative_crashes_list.append(current_crash_count)
            
    if not cumulative_crashes_list: return

    iterations = range(1, len(cumulative_crashes_list) + 1)
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, cumulative_crashes_list, label='Cumulative Unique Crashes', color='red', linewidth=2)
    plt.fill_between(iterations, cumulative_crashes_list, color='red', alpha=0.1)
    plt.title('Unique Crashes Found vs. Unique Inputs Discovered')
    plt.xlabel('Number of Unique Inputs Discovered')
    plt.ylabel('Cumulative Number of Unique Crashing Inputs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.savefig(PLOT_1_FILE)
    print(f"Saved {PLOT_1_FILE}")
    plt.close()

def run_tsne(data, n_samples):
    """运行 t-SNE 降维"""
    perplexity_value = max(5, n_samples - 1) if n_samples < 50 else min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, max_iter=1000, random_state=42)
    return tsne.fit_transform(data)

def plot_full_space(deduplicated_log, dtype_to_use, expected_size):
    """[图表 2] 绘制输入空间的 t-SNE 可视化"""
    all_data_list = []
    labels_list = []
    for entry in deduplicated_log:
        state_bytes = entry.get('mutate_state')
        if state_bytes is None or len(state_bytes) != expected_size: continue
        all_data_list.append(np.frombuffer(state_bytes, dtype=dtype_to_use))
        labels_list.append(1 if entry.get('did_crash', False) else 0)
        
    if not all_data_list: return

    all_data = np.vstack(all_data_list)
    labels = np.array(labels_list)
    
    # 如果数据太少，跳过 t-SNE 以避免错误
    if all_data.shape[0] < 5:
        print("Not enough data points for t-SNE visualization.")
        return

    tsne_results = run_tsne(all_data, all_data.shape[0])
    
    crashing_points = tsne_results[labels == 1]
    non_crashing_points = tsne_results[labels == 0]
    
    plt.figure(figsize=(12, 10))
    plt.scatter(non_crashing_points[:, 0], non_crashing_points[:, 1], c='blue', alpha=0.4, s=10, label=f'Non-Crashing ({non_crashing_points.shape[0]})')
    if crashing_points.shape[0] > 0:
        plt.scatter(crashing_points[:, 0], crashing_points[:, 1], c='red', alpha=0.8, s=15, label=f'Crashing ({crashing_points.shape[0]})')
    
    plt.title('t-SNE Visualization of Unique Explored Inputs')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(PLOT_2_FILE)
    print(f"Saved {PLOT_2_FILE}")
    plt.close()

def plot_generation_histogram(deduplicated_log):
    """[图表 3] 绘制故障代数直方图"""
    crash_generations = []
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            parent_depth = entry.get('parent_depth')
            if parent_depth is not None:
                crash_generations.append(parent_depth + 1)
            
    if not crash_generations: return

    generation_counts = Counter(crash_generations)
    max_gen = max(generation_counts.keys()) if generation_counts else 0
    generations = range(0, max_gen + 2)
    counts = [generation_counts.get(gen, 0) for gen in generations]
    
    print("\n--- Unique Crash Generation Stats ---")
    print(f"  Mean: {np.mean(crash_generations):.2f}")
    if len(crash_generations) > 0:
        print(f"  Median: {np.median(crash_generations)}")
        print(f"  Max: {np.max(crash_generations)}")

    plt.figure(figsize=(12, 7))
    plt.bar(generations, counts, color='red', alpha=0.7, zorder=3)
    plt.title('Histogram of Unique Crash Generations')
    plt.xlabel('Mutation Generation')
    plt.ylabel('Number of Unique Crashing Inputs')
    step = max(1, (max_gen // 20))
    plt.xticks(np.arange(0, max_gen + 2, step=step))
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.savefig(PLOT_3_FILE)
    print(f"Saved {PLOT_3_FILE}")
    plt.close()

def plot_crashes_over_time(deduplicated_log):
    """[图表 4] 绘制随时间变化的累计故障数"""
    crash_times = []
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            t = entry.get('elapsed_time')
            if t is not None: crash_times.append(t)
                
    if not crash_times: return
    crash_times.sort()
    crash_times_hours = [t / 3600.0 for t in crash_times]
    cumulative_counts = list(range(1, len(crash_times) + 1))
    
    plt.figure(figsize=(12, 7))
    plt.step(crash_times_hours, cumulative_counts, where='post', color='darkorange', linewidth=2, label='Cumulative Crashes')
    plt.fill_between(crash_times_hours, cumulative_counts, step='post', color='darkorange', alpha=0.1)
    plt.title('Cumulative Unique Crashes vs. Time')
    plt.xlabel('Time Elapsed (hours)')
    plt.ylabel('Cumulative Number of Unique Crashes')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.savefig(PLOT_4_FILE)
    print(f"Saved {PLOT_4_FILE}")
    plt.close()

def analyze_crash_statistics(deduplicated_log):
    """打印基本的故障统计信息 (基于去重后数据)"""
    unique_crash_inputs = 0
    unique_causing_seeds = set()
    
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            unique_crash_inputs += 1
            parent_seed = entry.get('seed_state')
            if parent_seed is not None:
                try:
                    seed_bytes = parent_seed.tobytes() if hasattr(parent_seed, 'tobytes') else np.array(parent_seed).tobytes()
                    unique_causing_seeds.add(seed_bytes)
                except Exception:
                    pass

    print(f"\n{'='*40}\n       Crash Analysis (Deep)\n{'='*40}")
    print(f"  Unique Crash Inputs: {unique_crash_inputs}")
    print(f"  Unique Source Seeds: {len(unique_causing_seeds)}")
    print(f"{'='*40}\n")

# --- QD-Fuzz 风格的行为多样性计算 ---

def calculate_behaviour_diversity(deduplicated_log, grid_size=(50, 50)):
    """
    计算基于 2D 网格 (Distance, Hull Angle) 的行为多样性 (覆盖率)。
    """
    print(f"\n{'='*40}\n       Behaviour Diversity Analysis (QD-Fuzz)\n{'='*40}")
    
    # 1. 提取行为特征 (BDs)
    dists = []
    angles = []
    is_crash_list = []
    
    found_bd = False
    
    for entry in deduplicated_log:
        d = entry.get('bd_distance')
        a = entry.get('bd_mean_angle')
        c = entry.get('did_crash', False)
        
        if d is not None and a is not None:
            dists.append(d)
            angles.append(a)
            is_crash_list.append(c)
            found_bd = True
            
    if not found_bd:
        print("Warning: No behavior descriptors ('bd_distance', 'bd_mean_angle') found in log.")
        print("Skipping behavior diversity calculation.")
        return

    dists = np.array(dists)
    angles = np.array(angles)
    
    # 2. 定义网格边界
    min_dist, max_dist = np.min(dists), np.max(dists)
    min_angle, max_angle = np.min(angles), np.max(angles)
    
    # 给边界加一点缓冲
    max_dist += 1e-5
    max_angle += 1e-5
    
    print(f"  Distance Range: [{min_dist:.2f}, {max_dist:.2f}]")
    print(f"  Angle Range:    [{min_angle:.2f}, {max_angle:.2f}]")
    
    # 3. 映射到网格
    filled_bins = set()
    filled_crash_bins = set()
    
    # 归一化并计算索引 [0, grid_size)
    if max_dist > min_dist:
        dist_indices = ((dists - min_dist) / (max_dist - min_dist) * grid_size[0]).astype(int)
    else:
        dist_indices = np.zeros_like(dists, dtype=int)
        
    if max_angle > min_angle:
        angle_indices = ((angles - min_angle) / (max_angle - min_angle) * grid_size[1]).astype(int)
    else:
        angle_indices = np.zeros_like(angles, dtype=int)
    
    # 安全裁剪索引范围
    dist_indices = np.clip(dist_indices, 0, grid_size[0] - 1)
    angle_indices = np.clip(angle_indices, 0, grid_size[1] - 1)
    
    for i in range(len(dists)):
        bin_id = (dist_indices[i], angle_indices[i])
        filled_bins.add(bin_id)
        if is_crash_list[i]:
            filled_crash_bins.add(bin_id)
            
    # 4. 输出报告
    total_bins = grid_size[0] * grid_size[1]
    print(f"  Behaviour Coverage (Total Filled Bins): {len(filled_bins)} / {total_bins} ({len(filled_bins)/total_bins:.2%})")
    print(f"  Fault Diversity (Total Crash Bins):     {len(filled_crash_bins)} (Unique crash types in behavior space)")
    print(f"{'='*40}\n")
    
    # 5. 绘制覆盖率热力图
    heatmap = np.zeros(grid_size)
    for i in range(len(dists)):
        heatmap[dist_indices[i], angle_indices[i]] += 1
        
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log1p(heatmap).T, origin='lower', aspect='auto', cmap='viridis', 
               extent=[min_dist, max_dist, min_angle, max_angle])
    plt.colorbar(label='Log(Count)')
    plt.title(f'Behaviour Space Coverage (Filled Bins: {len(filled_bins)})')
    plt.xlabel('Distance')
    plt.ylabel('Mean Hull Angle')
    plt.savefig(PLOT_5_FILE)
    print(f"Saved {PLOT_5_FILE}")
    plt.close()

def main():
    original_log_data = load_data(LOG_FILE)
    if not original_log_data: 
        print("Failed to load log data.")
        return
        
    deduplicated_log, dtype, expected_size = deduplicate_log(original_log_data)
    if not deduplicated_log: 
        print("Log data is empty or invalid.")
        return
    
    # [新增] 打印去重前后统计数据
    print_data_statistics(original_log_data, deduplicated_log)

    # 运行绘图逻辑
    plot_crash_trend(deduplicated_log)
    #plot_full_space(deduplicated_log, dtype, expected_size)
    #plot_generation_histogram(deduplicated_log)
    plot_crashes_over_time(deduplicated_log)
    #analyze_crash_statistics(deduplicated_log)
    
    # 运行行为多样性分析
    #calculate_behaviour_diversity(deduplicated_log)

    print("\nAll analysis and plotting completed.")

if __name__ == "__main__":
    main()