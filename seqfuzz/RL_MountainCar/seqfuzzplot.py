import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # 新增引用
from sklearn.manifold import TSNE
from collections import Counter
import os
import time

# --- 1. 配置 (已更新) ---
# 更新为您要分析的文件名
LOG_FILE = 'all_run_seeds_0.pkl' 
PLOT_1_FILE = 'crashes_over_unique_inputs.png'
PLOT_2_FILE = 'full_input_space_tsne.png'
PLOT_3_FILE = 'crash_generation_histogram.png'
PLOT_4_FILE = '1_crash_discovery_over_time.png' # 新增：随时间变化的崩溃图文件名

# --- 2. 核心辅助函数 (已修改) ---

def load_data(file_path):
    """加载 .pkl 文件"""
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        print(f"请确保此脚本与 '{file_path}' 位于同一文件夹中。")
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
    (已修改)
    根据 'state' (原 'mutate_state') 对日志进行去重，只保留第一次出现的条目。
    """
    print("正在根据 'state' 对日志进行去重（保留首次出现）...")
    
    seen_states = set()
    deduplicated_log = []
    dtype_to_use = None
    expected_size = 0
    
    # 动态检测 dtype
    # 假设 'state' 仍然是15个元素
    int32_size = 15 * np.dtype(np.int32).itemsize # 15 * 4 = 60
    int64_size = 15 * np.dtype(np.int64).itemsize # 15 * 8 = 120

    for entry in original_log_data:
        # (修改) 使用 'state'
        state = entry.get('state') 
        if state is None:
            continue
            
        try:
            state_bytes = state.tobytes()
        except AttributeError:
            # (修改) 更新警告信息
            print("警告: 发现非Numpy数组的 'state'，跳过。")
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
                # 尝试从数组形状动态推断（如果不是15维）
                expected_dim = state.shape
                print(f"警告: 无法识别的字节大小: {len(state_bytes)} 字节。")
                print(f"       将从数组形状 {expected_dim} 和 dtype {state.dtype} 动态设置。")
                dtype_to_use = state.dtype
                expected_size = len(state_bytes)
                
                # 更新 int32_size 和 int64_size 以匹配，防止后续检测失败
                if state.dtype == np.int32:
                    int32_size = expected_size
                elif state.dtype == np.int64:
                    int64_size = expected_size

        
        if len(state_bytes) != expected_size:
            continue
            
        # 核心去重逻辑
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            
            # 为了节省内存，我们在去重日志中存储字节，而不是Numpy数组
            entry_copy = entry.copy() 
            # (修改) 使用 'state'
            entry_copy['state'] = state_bytes 
            deduplicated_log.append(entry_copy)

    # (修改) 更新日志信息
    print(f"去重完成。总共找到 {len(deduplicated_log)} 个独特的 'state'。")
    
    if dtype_to_use is None:
         # (修改) 更新日志信息
        print("错误：未能从日志中检测到任何有效的 'state'。")
        return None, None, 0

    return deduplicated_log, dtype_to_use, expected_size


# --- 3. 图表1：崩溃趋势图 (已修改) ---

def plot_crash_trend(deduplicated_log):
    """
    (已修改)
    绘制随发现的“独特输入”数量而变化的“独特崩溃”累积数量。
    使用 'crashed' (原 'did_crash')
    """
    print(f"\n[图表 1] 正在计算崩溃趋势...")
    
    cumulative_crashes_list = []
    current_crash_count = 0
    
    for i, entry in enumerate(deduplicated_log):
        # (修改) 使用 'crashed'
        if entry.get('crashed', False): 
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


# --- 4. 图表2：t-SNE 空间图 (已修改) ---

def run_tsne(data, n_samples):
    """
    (保持不变)
    对 (N, D) 的数据运行 t-SNE 降维到 2 维 (D可能是15)
    """
    if n_samples < 50:
        print(f"数据点太少 ({n_samples}个)，t-SNE 可能效果不佳。")
        perplexity_value = max(5, n_samples - 1)
    else:
        perplexity_value = min(30, n_samples - 1)
        
    print(f"正在对 {n_samples} 个独特的输入（{data.shape[1]}维）运行 t-SNE (Perplexity={perplexity_value})...")
    
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
    (已修改)
    绘制 t-SNE 结果的 2D 散点图，用两种颜色区分
    使用 'state' (原 'mutate_state') 和 'crashed' (原 'did_crash')
    """
    print(f"\n[图表 2] 正在准备 t-SNE 数据...")
    
    all_data_list = []
    labels_list = []
    
    for entry in deduplicated_log:
        # (修改) 使用 'state'
        state_bytes = entry.get('state') 
        if state_bytes is None or len(state_bytes) != expected_size:
            continue
            
        all_data_list.append(np.frombuffer(state_bytes, dtype=dtype_to_use))
        # (修改) 使用 'crashed'
        labels_list.append(1 if entry.get('crashed', False) else 0) # 1 = Crash, 0 = No Crash
        
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


# --- 5. 图表3：崩溃代数直方图 (已修改) ---

def plot_generation_histogram(deduplicated_log):
    """
    (已修改)
    绘制 *独特崩溃输入* 的代数直方图。
    此版本直接从日志中读取 'generation'，不再使用 'parent_depth'。
    """
    print(f"\n[图表 3] 正在分析崩溃代数...")
    
    crash_generations = []

    for entry in deduplicated_log:
        # (修改) 使用 'crashed'
        if entry.get('crashed', False):
            # 这是一个在首次出现时就崩溃的独特输入
            
            # (修改) 直接读取 'generation'
            crash_generation = entry.get('generation') 
            
            if crash_generation is None:
                # (修改) 更新警告信息
                print("警告: 发现一个 crash 条目缺少 'generation'。可能来自旧版日志。跳过。")
                continue
            
            # (删除) 不再需要 'parent_depth + 1'
            crash_generations.append(crash_generation)
            
    print(f"分析完成。总共找到 {len(crash_generations)} 个独特的崩溃事件。")
    
    if not crash_generations:
        print("[图表 3] 未找到崩溃代数数据，无法绘图。")
        return

    generation_counts = Counter(crash_generations)
    
    # 确保我们至少绘制到第1代 (如果只有第0代种子，它们不会崩溃)
    max_gen = 0
    if generation_counts:
        max_gen = max(generation_counts.keys())
        
    # (修改) 从 0 开始，因为 generation 可能为 0
    generations = range(0, max_gen + 2) 
    counts = [generation_counts.get(gen, 0) for gen in generations]
    
    print("\n--- 独特崩溃代数统计 ---")
    print(f"  平均崩溃代数: {np.mean(crash_generations):.2f}")
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

def plot_crashes_over_time(deduplicated_log, total_samples_count):
    """
    (新增)
    绘制随时间（小时）变化的独特崩溃数量。
    依赖于日志条目中的 'crash_time' 字段。
    """
    print(f"\n[图表 4] 正在计算崩溃随时间的变化...")
    
    dedup_samples_count = len(deduplicated_log)
    
    # 提取 crash_time
    crash_times = []
    for entry in deduplicated_log:
        if entry.get('crashed', False):
            # 假设日志中有 'crash_time' 字段（单位通常为秒）
            t = entry.get('crash_time')
            if t is not None:
                crash_times.append(t)
            else:
                # 如果找不到时间，这里可以选择跳过或者记录警告
                # print("警告: 发现崩溃条目缺少 'crash_time'。")
                pass
                
    unique_crashes_count = len(crash_times)
    
    if not crash_times:
        print("[图表 4] 未找到包含 'crash_time' 的崩溃数据，跳过绘制。")
        return

    # 排序
    crash_times.sort()
    
    # 转换为小时
    times_in_hours = [t / 3600.0 for t in crash_times]
    counts = range(1, len(crash_times) + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(times_in_hours, counts, color='#E64A19', linewidth=3, label='Unique Crashes')
    plt.fill_between(times_in_hours, counts, color='#E64A19', alpha=0.1)
    
    plt.title('Crash Discovery Over Time', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Time (Hours)', fontsize=14, labelpad=10)
    plt.ylabel('Cumulative Crashes', fontsize=14, labelpad=10)
    
    # 设置Y轴为整数刻度
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 统计信息文本框
    stats_text = (
        f"$\\bf{{Statistics}}$\n"
        f"Total Samples: {total_samples_count}\n"
        f"Dedup. Samples: {dedup_samples_count}\n"
        f"Unique Crashes: {unique_crashes_count}"
    )
    
    # 文本框样式
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=13,
                   verticalalignment='top', horizontalalignment='left', bbox=props)
    
    try:
        plt.savefig(PLOT_4_FILE, dpi=300)
        print(f"[图表 4] 已保存到: {PLOT_4_FILE}")
    except Exception as e:
        print(f"[图表 4] 保存图表时出错: {e}")
    plt.close()


# --- 7. 主函数 (已更新) ---

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
    # 注意：这里我们传入原始日志的长度用于统计展示
    plot_crashes_over_time(deduplicated_log, len(original_log_data))
        
    print("\n所有分析和绘图已完成。")

if __name__ == "__main__":
    main()