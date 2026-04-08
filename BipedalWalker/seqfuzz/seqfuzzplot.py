import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import os
import time

# --- 1. 配置 ---
LOG_FILE = 'all_run_seeds_0.pkl' 

PLOT_1_FILE = 'crashes_over_unique_inputs.png'
PLOT_2_FILE = 'full_input_space_tsne.png'
PLOT_3_FILE = 'crash_generation_histogram.png'
PLOT_4_FILE = 'crashes_over_time.png'
PLOT_5_FILE = 'behaviour_coverage_heatmap.png' # [新增] 热力图文件

# --- 2. 核心辅助函数 ---

def load_data(file_path):
    """加载日志文件"""
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        print(f"请确保此脚本与 '{file_path}' 位于同一文件夹中，或修改脚本中的 LOG_FILE 路径。")
        return None
    
    try:
        print(f"正在从 {file_path} 加载原始日志数据...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"原始日志加载完成，总共 {len(data)} 条记录。")
        return data
    except Exception as e:
        print(f"加载 pickle 文件时出错: {e}")
        return None

def deduplicate_log(original_log_data):
    """
    根据 'state' 对日志进行去重，只保留第一次出现的条目。
    注意：这里的字段名已适配新的 enjoy.py 输出 ('state' 而非 'mutate_state')。
    """
    print("正在根据 'state' 对日志进行去重（保留首次出现）...")
    
    seen_states = set()
    deduplicated_log = []
    dtype_to_use = None
    expected_size = 0
    
    # 动态检测 dtype (兼容 int32 和 int64)
    int32_size = 15 * np.dtype(np.int32).itemsize # 60
    int64_size = 15 * np.dtype(np.int64).itemsize # 120

    for entry in original_log_data:
        state = entry.get('state')
        if state is None:
            continue
            
        try:
            state_bytes = state.tobytes()
        except AttributeError:
            print("警告: 发现非Numpy数组的 state，跳过。")
            continue
            
        if dtype_to_use is None:
            if len(state_bytes) == int32_size:
                print("检测到数据类型为 np.int32 (4 字节)")
                dtype_to_use = np.int32
                expected_size = int32_size
            elif len(state_bytes) == int64_size:
                print("检测到数据类型为 np.int64 (8 字节)")
                dtype_to_use = np.int64
                expected_size = int64_size
            else:
                print(f"错误: 无法识别的字节大小: {len(state_bytes)} 字节。跳过。")
                continue 
        
        if len(state_bytes) != expected_size:
            continue
            
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            entry_copy = entry.copy() 
            entry_copy['state_bytes'] = state_bytes 
            deduplicated_log.append(entry_copy)

    print(f"去重完成。总共找到 {len(deduplicated_log)} 个独特的 'state'。")
    
    if dtype_to_use is None:
        print("错误：未能从日志中检测到任何有效的 'state'。")
        return None, None, 0

    return deduplicated_log, dtype_to_use, expected_size


# --- 3. 图表1：崩溃趋势图 ---

def plot_crash_trend(deduplicated_log):
    print(f"\n[图表 1] 正在计算崩溃趋势...")
    cumulative_crashes_list = []
    current_crash_count = 0
    for i, entry in enumerate(deduplicated_log):
        if entry.get('crashed', False):
            current_crash_count += 1
        cumulative_crashes_list.append(current_crash_count)
            
    if not cumulative_crashes_list:
        print("[图表 1] 未找到可绘制的崩溃趋势数据。")
        return

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
    try:
        plt.savefig(PLOT_1_FILE)
        print(f"[图表 1] 已保存到: {PLOT_1_FILE}")
    except Exception as e:
        print(f"[图表 1] 保存图表时出错: {e}")
    plt.close()


# --- 4. 图表2：t-SNE 空间图 ---

def run_tsne(data, n_samples):
    if n_samples < 50:
        perplexity_value = max(5, n_samples - 1)
    else:
        perplexity_value = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity_value, max_iter=1000, random_state=42)
    return tsne.fit_transform(data)

def plot_full_space(deduplicated_log, dtype_to_use, expected_size):
    print(f"\n[图表 2] 正在准备 t-SNE 数据...")
    all_data_list = []
    labels_list = []
    for entry in deduplicated_log:
        state_bytes = entry.get('state_bytes') 
        if state_bytes is None or len(state_bytes) != expected_size: continue
        all_data_list.append(np.frombuffer(state_bytes, dtype=dtype_to_use))
        labels_list.append(1 if entry.get('crashed', False) else 0) 
        
    if not all_data_list:
        print("[图表 2] 未找到可用于 t-SNE 的数据。")
        return

    all_data = np.vstack(all_data_list)
    labels = np.array(labels_list)
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
    try:
        plt.savefig(PLOT_2_FILE)
        print(f"[图表 2] 已保存到: {PLOT_2_FILE}")
    except Exception as e:
        print(f"[图表 2] 保存图表时出错: {e}")
    plt.close()


# --- 5. 图表3：崩溃代数直方图 ---

def plot_generation_histogram(deduplicated_log):
    print(f"\n[图表 3] 正在分析崩溃代数...")
    crash_generations = []
    for entry in deduplicated_log:
        if entry.get('crashed', False):
            gen = entry.get('generation') 
            if gen is not None: crash_generations.append(gen)
            
    if not crash_generations:
        print("[图表 3] 未找到崩溃代数数据，无法绘图。")
        return

    generation_counts = Counter(crash_generations)
    max_gen = max(generation_counts.keys()) if generation_counts else 0
    generations = range(0, max_gen + 2)
    counts = [generation_counts.get(gen, 0) for gen in generations]
    
    print("\n--- 独特崩溃代数统计 ---")
    print(f"  平均: {np.mean(crash_generations):.2f}")

    plt.figure(figsize=(12, 7))
    plt.bar(generations, counts, color='red', alpha=0.7, zorder=3)
    plt.title('Histogram of Unique Crash Generations')
    plt.xlabel('Mutation Generation')
    plt.ylabel('Number of Unique Crashing Inputs')
    step = max(1, (max_gen // 20))
    plt.xticks(np.arange(0, max_gen + 2, step=step))
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    try:
        plt.savefig(PLOT_3_FILE)
        print(f"[图表 3] 已保存到: {PLOT_3_FILE}")
    except Exception as e:
        print(f"[图表 3] 保存图表时出错: {e}")
    plt.close()


# --- 6. 图表4：崩溃随时间变化图 ---

def plot_crashes_over_time(deduplicated_log):
    print(f"\n[图表 4] 正在分析崩溃随时间的变化...")
    crash_times = []
    for entry in deduplicated_log:
        if entry.get('crashed', False):
            t = entry.get('timestamp')
            if t is not None: crash_times.append(t)
                
    if not crash_times:
        print("[图表 4] 未在崩溃数据中找到 'timestamp' 字段。")
        return

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
    try:
        plt.savefig(PLOT_4_FILE)
        print(f"[图表 4] 已保存到: {PLOT_4_FILE}")
    except Exception as e:
        print(f"[图表 4] 保存图表时出错: {e}")
    plt.close()

# --- [新增] 7. 图表5：QD-Fuzz 风格行为多样性分析 ---

def calculate_behaviour_diversity(deduplicated_log, grid_size=(50, 50)):
    """
    [新增] 计算基于 2D 网格 (Distance, Hull Angle) 的行为多样性 (覆盖率)
    """
    print(f"\n{'='*40}\n       Behaviour Diversity Analysis (QD-Fuzz)\n{'='*40}")
    
    dists = []
    angles = []
    is_crash_list = []
    found_bd = False
    
    for entry in deduplicated_log:
        d = entry.get('bd_distance')
        a = entry.get('bd_mean_angle')
        c = entry.get('crashed', False)
        
        if d is not None and a is not None:
            dists.append(d)
            angles.append(a)
            is_crash_list.append(c)
            found_bd = True
            
    if not found_bd:
        print("警告: 日志中未找到行为描述符 ('bd_distance', 'bd_mean_angle')。")
        print("请确保运行了包含 BD 计算的新版 enjoy.py。")
        return

    dists = np.array(dists)
    angles = np.array(angles)
    
    min_dist, max_dist = np.min(dists), np.max(dists)
    min_angle, max_angle = np.min(angles), np.max(angles)
    
    # 增加微小缓冲
    max_dist += 1e-5
    max_angle += 1e-5
    
    print(f"  Distance Range: [{min_dist:.2f}, {max_dist:.2f}]")
    print(f"  Angle Range:    [{min_angle:.2f}, {max_angle:.2f}]")
    
    filled_bins = set()
    filled_crash_bins = set()
    
    if max_dist > min_dist:
        dist_indices = ((dists - min_dist) / (max_dist - min_dist) * grid_size[0]).astype(int)
    else:
        dist_indices = np.zeros_like(dists, dtype=int)
        
    if max_angle > min_angle:
        angle_indices = ((angles - min_angle) / (max_angle - min_angle) * grid_size[1]).astype(int)
    else:
        angle_indices = np.zeros_like(angles, dtype=int)
    
    dist_indices = np.clip(dist_indices, 0, grid_size[0] - 1)
    angle_indices = np.clip(angle_indices, 0, grid_size[1] - 1)
    
    for i in range(len(dists)):
        bin_id = (dist_indices[i], angle_indices[i])
        filled_bins.add(bin_id)
        if is_crash_list[i]:
            filled_crash_bins.add(bin_id)
            
    total_bins = grid_size[0] * grid_size[1]
    print(f"  Behaviour Coverage (Total Filled Bins): {len(filled_bins)} / {total_bins} ({len(filled_bins)/total_bins:.2%})")
    print(f"  Fault Diversity (Total Crash Bins):     {len(filled_crash_bins)} (Unique crash types in behavior space)")
    print(f"{'='*40}\n")
    
    # 绘制热力图
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
    try:
        plt.savefig(PLOT_5_FILE)
        print(f"[图表 5] 已保存到: {PLOT_5_FILE}")
    except Exception as e:
        print(f"[图表 5] 保存图表时出错: {e}")
    plt.close()

# --- 主函数 ---

def main():
    original_log_data = load_data(LOG_FILE)
    if not original_log_data: return

    deduplicated_log, dtype, expected_size = deduplicate_log(original_log_data)
    if not deduplicated_log:
        print("未能从日志中提取任何有效数据。退出。")
        return
    
    unique_crashes_count = sum(1 for entry in deduplicated_log if entry.get('crashed', False))
    print(f"\n[统计] 发现的独特 Crash 输入总数: {unique_crashes_count}")   
    
    #plot_crash_trend(deduplicated_log)
    #plot_full_space(deduplicated_log, dtype, expected_size)
    #plot_generation_histogram(deduplicated_log)
    #plot_crashes_over_time(deduplicated_log)
    
    # [新增] 运行行为多样性分析
    #calculate_behaviour_diversity(deduplicated_log)
        
    print("\n所有分析和绘图已完成。")

if __name__ == "__main__":
    main()