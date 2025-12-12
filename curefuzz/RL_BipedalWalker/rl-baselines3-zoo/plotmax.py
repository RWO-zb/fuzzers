import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import os
import time

# --- 1. 配置 ---
LOG_FILE = 'selection_log.pkl'
PLOT_1_FILE = 'crashes_over_unique_inputs.png'
PLOT_2_FILE = 'full_input_space_tsne.png'
PLOT_3_FILE = 'crash_generation_histogram.png'
PLOT_4_FILE = 'crashes_over_time.png'  # <--- 新增配置

# --- 2. 核心辅助函数 (保持不变) ---

def load_data(file_path):
    """加载 selection_log.pkl 文件"""
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        print("请确保此脚本与 'selection_log.pkl' 位于同一文件夹中。")
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
    (保持不变)
    根据 'mutate_state' 对日志进行去重，只保留第一次出现的条目。
    """
    print("正在根据 'mutate_state' 对日志进行去重（保留首次出现）...")
    
    seen_mutate_states = set()
    deduplicated_log = []
    dtype_to_use = None
    expected_size = 0
    
    # 动态检测 dtype
    int32_size = 15 * np.dtype(np.int32).itemsize # 15 * 4 = 60
    int64_size = 15 * np.dtype(np.int64).itemsize # 15 * 8 = 120

    for entry in original_log_data:
        state = entry.get('mutate_state')
        if state is None:
            continue
            
        try:
            state_bytes = state.tobytes()
        except AttributeError:
            print("警告: 发现非Numpy数组的 mutate_state，跳过。")
            continue
            
        # 第一次运行时检测 dtype
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
            
        # 核心去重逻辑
        if state_bytes not in seen_mutate_states:
            seen_mutate_states.add(state_bytes)
            
            # 为了节省内存，我们在去重日志中存储字节，而不是Numpy数组
            # 关键：entry.copy() 会保留 'parent_depth' 等其他元数据
            entry_copy = entry.copy() 
            entry_copy['mutate_state'] = state_bytes
            deduplicated_log.append(entry_copy)

    print(f"去重完成。总共找到 {len(deduplicated_log)} 个独特的 'mutate_state'。")
    
    if dtype_to_use is None:
        print("错误：未能从日志中检测到任何有效的 'mutate_state'。")
        return None, None, 0

    return deduplicated_log, dtype_to_use, expected_size


# --- 3. 图表1：崩溃趋势图 (保持不变) ---

def plot_crash_trend(deduplicated_log):
    """
    (保持不变)
    绘制随发现的“独特输入”数量而变化的“独特崩溃”累积数量。
    """
    print(f"\n[图表 1] 正在计算崩溃趋势...")
    
    cumulative_crashes_list = []
    current_crash_count = 0
    
    for i, entry in enumerate(deduplicated_log):
        if entry.get('did_crash', False):
            current_crash_count += 1
        
        cumulative_crashes_list.append(current_crash_count)
            
    if not cumulative_crashes_list:
        print("[图表 1] 未找到可绘制的崩溃趋势数据。")
        return

    print(f"正在绘制 {len(cumulative_crashes_list)} 个独特输入...")
    
    iterations = range(1, len(cumulative_crashes_list) + 1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, cumulative_crashes_list, label='Cumulative Unique Crashes', color='red', linewidth=2)
    plt.fill_between(iterations, cumulative_crashes_list, color='red', alpha=0.1)
    
    plt.title('Unique Crashes Found vs. Unique Inputs Discovered')
    plt.xlabel('Number of Unique Inputs Discovered (by First Appearance)')
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
    plt.close() # 释放内存


# --- 4. 图表2：t-SNE 空间图 (保持不变) ---

def run_tsne(data, n_samples):
    """
    (保持不变)
    对 (N, 15) 的数据运行 t-SNE 降维到 2 维
    """
    if n_samples < 50:
        print(f"数据点太少 ({n_samples}个)，t-SNE 可能效果不佳。")
        perplexity_value = max(5, n_samples - 1)
    else:
        perplexity_value = min(30, n_samples - 1)
        
    print(f"正在对 {n_samples} 个独特的输入（15维）运行 t-SNE (Perplexity={perplexity_value})...")
    
    start_time = time.time()
    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=perplexity_value,
        max_iter=1000, 
        random_state=42
    )
    tsne_results = tsne.fit_transform(data)
    
    end_time = time.time()
    print(f"t-SNE 运行完成，耗时: {end_time - start_time:.2f} 秒")
    return tsne_results

def plot_full_space(deduplicated_log, dtype_to_use, expected_size):
    """
    (保持不变)
    绘制 t-SNE 结果的 2D 散点图，用两种颜色区分
    """
    print(f"\n[图表 2] 正在准备 t-SNE 数据...")
    
    all_data_list = []
    labels_list = []
    
    for entry in deduplicated_log:
        state_bytes = entry.get('mutate_state')
        if state_bytes is None or len(state_bytes) != expected_size:
            continue
            
        all_data_list.append(np.frombuffer(state_bytes, dtype=dtype_to_use))
        labels_list.append(1 if entry.get('did_crash', False) else 0) # 1 = Crash, 0 = No Crash
        
    if not all_data_list:
        print("[图表 2] 未找到可用于 t-SNE 的数据。")
        return

    all_data = np.vstack(all_data_list)
    labels = np.array(labels_list)
    
    # 运行 t-SNE
    tsne_results = run_tsne(all_data, all_data.shape[0])
    
    # 绘制
    print(f"正在绘制 {tsne_results.shape[0]} 个数据点...")
    crashing_points = tsne_results[labels == 1]
    non_crashing_points = tsne_results[labels == 0]
    
    print(f"  非崩溃点: {non_crashing_points.shape[0]}")
    print(f"  崩溃点: {crashing_points.shape[0]}")
    
    plt.figure(figsize=(12, 10))
    plt.scatter(
        non_crashing_points[:, 0], non_crashing_points[:, 1], 
        c='blue', alpha=0.4, s=10, 
        label=f'Non-Crashing Inputs ({non_crashing_points.shape[0]})'
    )
    if crashing_points.shape[0] > 0:
        plt.scatter(
            crashing_points[:, 0], crashing_points[:, 1], 
            c='red', alpha=0.8, s=15, 
            label=f'Crashing Inputs ({crashing_points.shape[0]})'
        )
    
    plt.title('t-SNE Visualization of Unique Explored Inputs (by First Appearance)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    try:
        plt.savefig(PLOT_2_FILE)
        print(f"[图表 2] 已保存到: {PLOT_2_FILE}")
    except Exception as e:
        print(f"[图表 2] 保存图表时出错: {e}")
    plt.close() # 释放内存


# --- 5. 图表3：崩溃代数直方图 (保持不变) ---

def plot_generation_histogram(deduplicated_log):
    """
    (保持不变)
    绘制 *独特崩溃输入* 的代数直方图。
    """
    print(f"\n[图表 3] 正在分析崩溃代数...")
    
    crash_generations = []

    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            # 这是一个在首次出现时就崩溃的独特输入
            
            # 'parent_depth' 是在 enjoy_cure.py 中记录的父种子的代数
            parent_depth = entry.get('parent_depth') 
            
            if parent_depth is None:
                # 兼容旧日志或缺失数据的情况
                # print("警告: 发现一个 crash 条目缺少 'parent_depth'。")
                continue
            
            # crash 本身的代数 = 父代数 + 1
            crash_generation = parent_depth + 1
            crash_generations.append(crash_generation)
            
    print(f"分析完成。总共找到 {len(crash_generations)} 个独特的崩溃事件。")
    
    if not crash_generations:
        print("[图表 3] 未找到崩溃代数数据，无法绘图。")
        return

    generation_counts = Counter(crash_generations)
    
    # 确保我们至少绘制到第1代
    max_gen = 0
    if generation_counts:
        max_gen = max(generation_counts.keys())
        
    generations = range(0, max_gen + 2) # 从0开始，多显示一格
    counts = [generation_counts.get(gen, 0) for gen in generations]
    
    print("\n--- 独特崩溃代数统计 ---")
    print(f"  平均崩溃代数: {np.mean(crash_generations):.2f}")
    if len(crash_generations) > 0:
        print(f"  中位崩溃代数: {np.median(crash_generations)}")
        print(f"  最大崩溃代数: {np.max(crash_generations)}")

    plt.figure(figsize=(12, 7))
    plt.bar(generations, counts, color='red', alpha=0.7, zorder=3)
    
    plt.title('Histogram of Unique Crash Generations (by First Appearance)')
    plt.xlabel('Mutation Generation (Depth from Initial Seed)')
    plt.ylabel('Number of Unique Crashing Inputs')
    
    step = max(1, (max_gen // 20)) # 动态调整x轴刻度
    plt.xticks(np.arange(0, max_gen + 2, step=step))
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    try:
        plt.savefig(PLOT_3_FILE)
        print(f"[图表 3] 已保存到: {PLOT_3_FILE}")
    except Exception as e:
        print(f"[图表 3] 保存图表时出错: {e}")
    plt.close() # 释放内存


# --- 6. 图表4：崩溃随时间变化图 (新增) ---

def plot_crashes_over_time(deduplicated_log):
    """
    (新增)
    绘制独特崩溃随时间变化的累积曲线。
    依赖于日志条目中的 'elapsed_time' 字段。
    """
    print(f"\n[图表 4] 正在分析崩溃随时间的变化...")
    
    crash_times = []
    
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            # 尝试获取时间戳
            t = entry.get('elapsed_time')
            if t is not None:
                crash_times.append(t)
            else:
                # 如果是旧日志没有时间戳，这部分数据将被忽略
                pass
                
    if not crash_times:
        print("[图表 4] 未在崩溃数据中找到 'elapsed_time' 字段。无法绘制时间曲线。")
        print("提示：请确保您运行的是更新后的 enjoy_cure.py。")
        return

    # 确保按时间排序（理论上日志是顺序的，但去重后顺序可能取决于保留的是哪一个，通常是第一个）
    # 但去重逻辑保留的是原列表中的顺序，所以 sort 是为了保险
    crash_times.sort()
    
    # 构造累积数量
    cumulative_counts = list(range(1, len(crash_times) + 1))
    
    # 打印一些统计信息
    print(f"  共找到 {len(crash_times)} 个带有时间戳的独特崩溃。")
    print(f"  首次崩溃时间: {crash_times[0]:.2f} 秒")
    print(f"  最后崩溃时间: {crash_times[-1]:.2f} 秒")
    
    plt.figure(figsize=(12, 7))
    
    # 使用 step 图可以更清晰地显示离散的发现过程
    plt.step(crash_times, cumulative_counts, where='post', color='darkorange', linewidth=2, label='Cumulative Crashes')
    plt.fill_between(crash_times, cumulative_counts, step='post', color='darkorange', alpha=0.1)
    
    plt.title('Cumulative Unique Crashes vs. Time')
    plt.xlabel('Time Elapsed (seconds)')
    plt.ylabel('Cumulative Number of Unique Crashes')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 设置坐标轴范围更好看一点
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    
    try:
        plt.savefig(PLOT_4_FILE)
        print(f"[图表 4] 已保存到: {PLOT_4_FILE}")
    except Exception as e:
        print(f"[图表 4] 保存图表时出错: {e}")
    plt.close()


# --- 7. 主函数 ---

def main():
    # 1. 加载原始数据
    original_log_data = load_data(LOG_FILE)
    if not original_log_data:
        return

    # 2. 对日志进行去重
    deduplicated_log, dtype, expected_size = deduplicate_log(original_log_data)
    if not deduplicated_log:
        print("未能从日志中提取任何有效数据。退出。")
        return
        
    # 3. 运行图表 1
    plot_crash_trend(deduplicated_log)
    
    # 4. 运行图表 2
    plot_full_space(deduplicated_log, dtype, expected_size)
    
    # 5. 运行图表 3
    plot_generation_histogram(deduplicated_log)
    
    # 6. 运行图表 4 (新增)
    plot_crashes_over_time(deduplicated_log)
        
    print("\n所有分析和绘图已完成。")

if __name__ == "__main__":
    main()