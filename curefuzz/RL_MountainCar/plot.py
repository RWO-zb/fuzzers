import pickle
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE # t-SNE 已不再需要
from collections import Counter
import os
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# --- 1. 配置 ---
LOG_FILE = 'D:\\code\\fuzzers\\curefuzz\\mountaincar\\results\\11_22_2025_17_26_11_seed_0\\selection_log.pkl'
PLOT_1_FILE = '1_crashes_over_time.png'
PLOT_2_FILE = '2_state_space_visualization.png' # 文件名已更新
PLOT_3_FILE = '3_crash_generation_hist_filtered.png'

# 设置全局绘图风格
sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

# --- 2. 核心辅助函数 ---

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"日志加载完成，共 {len(data)} 条记录。")
        return data
    except Exception as e:
        print(f"加载出错: {e}")
        return None

def deduplicate_log(original_log_data):
    print("正在去重并解析数据...")
    seen_mutate_states = set()
    deduplicated_log = []
    
    # MountainCar: 2个 float64 (16 bytes) 或 float32 (8 bytes)
    size_2_float64 = 2 * np.dtype(np.float64).itemsize 
    size_2_float32 = 2 * np.dtype(np.float32).itemsize 
    dtype_to_use = None
    expected_size = 0

    for entry in original_log_data:
        state = entry.get('mutate_state')
        if state is None: continue
        try:
            state_bytes = state.tobytes()
        except AttributeError: continue
            
        if dtype_to_use is None:
            if len(state_bytes) == size_2_float64:
                dtype_to_use = np.float64
                expected_size = size_2_float64
            elif len(state_bytes) == size_2_float32:
                dtype_to_use = np.float32
                expected_size = size_2_float32
            else:
                dtype_to_use = np.float64
                expected_size = len(state_bytes)
        
        if len(state_bytes) != expected_size: continue
            
        if state_bytes not in seen_mutate_states:
            seen_mutate_states.add(state_bytes)
            entry_copy = entry.copy()
            entry_copy['mutate_state_bytes'] = state_bytes 
            deduplicated_log.append(entry_copy)

    print(f"去重完成。共 {len(deduplicated_log)} 个独特输入。")
    return deduplicated_log, dtype_to_use

# --- 3. 图表1：Crash 数量随时间变化 (小时) ---

def plot_crashes_over_time(deduplicated_log):
    print(f"\n[图表 1] 绘制 Crash 趋势图...")
    crash_times = []
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            t = entry.get('crash_time')
            if t is not None: crash_times.append(t)
    
    if not crash_times:
        print("[图表 1] 无数据。")
        return

    crash_times.sort()
    # 转换为小时
    crash_times_hours = [t / 3600.0 for t in crash_times]
    cumulative_counts = range(1, len(crash_times) + 1)
    
    plt.figure(figsize=(12, 7))
    line_color = "#D32F2F"
    plt.plot(crash_times_hours, cumulative_counts, color=line_color, linewidth=3, label='Cumulative Crashes')
    plt.fill_between(crash_times_hours, cumulative_counts, color=line_color, alpha=0.1)
    
    plt.title('Cumulative Unique Crashes vs. Time', fontweight='bold', pad=20)
    plt.xlabel('Time Elapsed (Hours)', labelpad=10)
    plt.ylabel('Unique Crashes Found', labelpad=10)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    sns.despine()
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOT_1_FILE, dpi=300)
    plt.close()

# --- 4. 图表2：原始物理状态空间分布 (Position vs Velocity) ---
# 替换了原来的 t-SNE 函数

def plot_state_space_visualization(deduplicated_log, dtype):
    print(f"\n[图表 2] 绘制物理状态空间 (Position vs Velocity)...")
    
    all_data, labels = [], []
    for entry in deduplicated_log:
        state_bytes = entry.get('mutate_state_bytes')
        if state_bytes is None: continue
        state_vec = np.frombuffer(state_bytes, dtype=dtype)
        all_data.append(state_vec)
        labels.append(1 if entry.get('did_crash', False) else 0)
            
    if not all_data: return
    X = np.vstack(all_data) # X[:,0] = Position, X[:,1] = Velocity
    y = np.array(labels)
    
    plt.figure(figsize=(12, 8))
    
    # 1. 绘制正常点 (Safe States)
    # 使用深青色作为背景，透明度低，展示探索范围
    mask_normal = (y == 0)
    plt.scatter(X[mask_normal, 0], X[mask_normal, 1], 
                c='#455A64', s=20, alpha=0.2, label='Safe States', edgecolors='none')
    
    # 2. 绘制 Crash 点 (Crash States)
    # 使用鲜红色，不透明，强调危险区域
    mask_crash = (y == 1)
    if np.sum(mask_crash) > 0:
        plt.scatter(X[mask_crash, 0], X[mask_crash, 1], 
                    c='#D32F2F', s=50, alpha=0.9, marker='x', label='Crash States', linewidth=1.5)
        
    plt.title('State Space Coverage & Crash Distribution', fontweight='bold', pad=20)
    plt.xlabel('Car Position', labelpad=10)
    plt.ylabel('Car Velocity', labelpad=10)
    
    # 设置 MountainCar 的典型物理边界，使图表更专业
    plt.xlim(-1.2, 0.6)
    plt.ylim(-0.07, 0.07)
    
    # 添加图例和去边框
    sns.despine()
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(PLOT_2_FILE, dpi=300)
    print(f"  -> 已保存: {PLOT_2_FILE}")
    plt.close()

# --- 5. 图表3：Crash 变异代数直方图 (统计与过滤版) ---

def plot_generation_histogram(deduplicated_log):
    print(f"\n[图表 3] 分析代数分布并过滤离群点...")
    
    crash_gens = []
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            parent_depth = entry.get('parent_depth', -1)
            gen = parent_depth + 1 if parent_depth is not None else 0
            crash_gens.append(gen)
            
    if not crash_gens:
        print("[图表 3] 无 Crash 数据。")
        return
    
    # 统计数据
    mean_gen = np.mean(crash_gens)
    median_gen = np.median(crash_gens)
    max_gen = np.max(crash_gens)
    
    print(f"  统计结果: Mean={mean_gen:.2f}, Median={median_gen}, Max={max_gen}")

    # 过滤离群点 (IQR)
    q1 = np.percentile(crash_gens, 25)
    q3 = np.percentile(crash_gens, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    if iqr == 0: upper_bound = max_gen
    
    filtered_gens = [g for g in crash_gens if g <= upper_bound]
    filtered_count = len(crash_gens) - len(filtered_gens)
    
    # 绘图数据
    gen_counts = Counter(filtered_gens)
    if not gen_counts: return

    sorted_gens = sorted(gen_counts.keys())
    x_vals = sorted_gens
    y_vals = [gen_counts[g] for g in x_vals]
    
    plt.figure(figsize=(12, 7))
    
    norm = plt.Normalize(0, max(y_vals))
    colors = sns.color_palette("YlOrRd", as_cmap=True)(norm(y_vals))

    bars = plt.bar(x_vals, y_vals, color=colors, edgecolor='#333333', linewidth=0.8, width=0.7, alpha=0.9)
    
    plt.title('Distribution of Crashes by Mutation Generation', fontweight='bold', pad=20)
    plt.xlabel('Mutation Generation (Depth)', labelpad=10)
    plt.ylabel('Count of Unique Crashes', labelpad=10)
    
    plt.xticks(x_vals)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    sns.despine()
    plt.grid(axis='x', visible=False)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                     ha='center', va='bottom', fontsize=12, fontweight='bold', color='#444444')

    # 统计信息框
    stats_text = (
        f"$\\bf{{Statistics\ (Full\ Data)}}$\n"
        f"Mean Generation: {mean_gen:.2f}\n"
        f"Median Generation: {int(median_gen)}\n"
        f"Max Generation: {int(max_gen)}\n"
        f"----------------\n"
        f"Display Limit: ≤ {int(upper_bound)}\n"
        f"(Outliers Hidden: {filtered_count})"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(PLOT_3_FILE, dpi=300)
    print(f"  -> 已保存: {PLOT_3_FILE}")
    plt.close()

# --- 主程序 ---

def main():
    data = load_data(LOG_FILE)
    if not data: return
    
    dedup_log, dtype = deduplicate_log(data)
    if not dedup_log: return
        
    plot_crashes_over_time(dedup_log)
    plot_state_space_visualization(dedup_log, dtype) # 使用新的物理空间绘图
    plot_generation_histogram(dedup_log)
    
    print("\n所有高分辨率美化图表已生成。")

if __name__ == "__main__":
    main()