import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter  # <--- 新增引用

#input file 
LOG_FILE = 'all_run_seeds_723.pkl'

#output file 
PLOT_BD_CURVE = 'seqfuzz_behaviour_diversity_curve.png'
PLOT_FD_CURVE = 'seqfuzz_fault_diversity_curve.png'
PLOT_SC_CURVE = 'seqfuzz_state_coverage_curve.png'
PLOT_SEED_DIST = 'seqfuzz_seed_crash_distribution.png' # <--- 新增输出文件

GRID_SIZE = (50, 50)

def load_data(file_path):
    print(f"Loading log data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(f"Loaded {len(data)} entries.")
        return data

def get_bin_index(value, min_val, max_val, grid_size_dim):
    if max_val <= min_val:
        return 0
    idx = int((value - min_val) / (max_val - min_val) * grid_size_dim)
    return min(max(idx, 0), grid_size_dim - 1)

def calculate_cumulative_trends(log_data):

    all_dists = []
    all_angles = []
    
    for entry in log_data:
        d = entry.get('bd_distance')
        a = entry.get('bd_mean_angle')
        if d is not None and a is not None:
            all_dists.append(d)
            all_angles.append(a)
            
    if not all_dists:
        print("Warning: No behaviour data (bd_distance/bd_mean_angle) found in log.")
        return None

    min_dist, max_dist = min(all_dists), max(all_dists) + 1e-5
    min_angle, max_angle = min(all_angles), max(all_angles) + 1e-5
    
    print(f"Global Range - Dist: [{min_dist:.2f}, {max_dist:.2f}], Angle: [{min_angle:.2f}, {max_angle:.2f}]")

    bd_filled_bins = set()     
    fd_crash_bins = set()      
    unique_states = set()     
    
    bd_trend = []
    fd_trend = []
    sc_trend = []
    
    for i, entry in enumerate(log_data):
        state = entry.get('state')
        if state is not None:
            state_bytes = state.tobytes() if hasattr(state, 'tobytes') else np.array(state).tobytes()
            unique_states.add(state_bytes)      
        sc_trend.append(len(unique_states))

        d = entry.get('bd_distance')
        a = entry.get('bd_mean_angle')
        is_crash = entry.get('crashed', False)
        
        if d is not None and a is not None:
            d_idx = get_bin_index(d, min_dist, max_dist, GRID_SIZE[0])
            a_idx = get_bin_index(a, min_angle, max_angle, GRID_SIZE[1])
            bin_loc = (d_idx, a_idx)
            bd_filled_bins.add(bin_loc)
            if is_crash:
                fd_crash_bins.add(bin_loc)
        
        bd_trend.append(len(bd_filled_bins))
        fd_trend.append(len(fd_crash_bins))
        
    return {
        'x_axis': range(1, len(log_data) + 1), 
        'bd_trend': bd_trend,
        'fd_trend': fd_trend,
        'sc_trend': sc_trend
    }

def plot_curve(x, y, title, ylabel, filename, color):
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, y, linewidth=2, color=color, label='SeqFuzz')
    plt.fill_between(x, y, color=color, alpha=0.1)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Number of Test Cases', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0, right=len(x))
    plt.ylim(bottom=0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")
    plt.close()

# --- 新增功能：统计并绘制导致 Crash 的初始种子分布 ---
def plot_crash_seed_distribution(log_data, filename):
    print("\nProcessing Crash Seed Distribution...")
    
    # 1. 筛选出所有 Crash 的条目
    crash_entries = [d for d in log_data if d.get('crashed', False)]
    
    if not crash_entries:
        print("No crashes found in log data. Skipping seed distribution plot.")
        return

    # 2. 提取 root_id
    # 注意：如果使用旧日志运行，可能不存在 root_id，这里做个兼容处理
    root_ids = []
    missing_id_count = 0
    for d in crash_entries:
        rid = d.get('root_id')
        if rid is not None:
            root_ids.append(rid)
        else:
            missing_id_count += 1
            
    if missing_id_count > 0:
        print(f"Warning: {missing_id_count} crash entries exist but lack 'root_id'. Please run new experiments with updated enjoy.py.")
    
    if not root_ids:
        print("No valid root_ids found in crash entries.")
        return

    # 3. 统计频率
    counts = Counter(root_ids)
    
    # 统计有多少个不同的初始种子
    unique_seeds_count = len(counts)
    total_crashes = len(root_ids)
    
    print(f"Found {total_crashes} total crashes across {unique_seeds_count} distinct initial seeds.")

    # 4. 准备绘图数据 (按 Seed ID 排序，以便观察 ID 分布)
    sorted_ids = sorted(counts.keys())
    sorted_counts = [counts[k] for k in sorted_ids]
    
    # 将 ID 转换为字符串以便作为 X 轴标签（如果 ID 是整数）
    x_labels = [str(k) for k in sorted_ids]

    # 5. 绘制柱状图
    plt.figure(figsize=(12, 6))
    
    # 如果数据点太多，可能需要调整柱子宽度或只显示 Top N，这里默认全部显示
    bars = plt.bar(x_labels, sorted_counts, color='#e67e22', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    plt.title(f'Distribution of Crashes by Initial Seed (Root ID)\nTotal Unique Seeds Causing Crash: {unique_seeds_count}', fontsize=14, fontweight='bold')
    plt.xlabel('Initial Seed ID (Root ID)', fontsize=12)
    plt.ylabel('Number of Crashes Found', fontsize=12)
    
    # 如果 X 轴标签太多（超过30个），为了美观，稀疏化显示或旋转
    if len(x_labels) > 30:
        plt.xticks(rotation=90, fontsize=8)
        # 仅显示部分刻度以防重叠
        n = len(x_labels)
        step = max(1, n // 50)
        plt.xticks(range(0, n, step), [x_labels[i] for i in range(0, n, step)])
    else:
        plt.xticks(rotation=45)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    print(f"Saved crash distribution plot to {filename}")
    plt.close()

def main():
    log_data = load_data(LOG_FILE)
    
    if not log_data:
        return

    trends = calculate_cumulative_trends(log_data)
    
    if trends:
        x = trends['x_axis']

        plot_curve(
            x, trends['bd_trend'], 
            title='Behaviour Diversity Growth', 
            ylabel='Cumulative Covered Bins (Behaviour)', 
            filename=PLOT_BD_CURVE,
            color='#9b59b6' 
        )
        
        plot_curve(
            x, trends['fd_trend'], 
            title='Fault Diversity Growth', 
            ylabel='Cumulative Covered Crash Bins', 
            filename=PLOT_FD_CURVE,
            color='#e74c3c' 
        )
        
        plot_curve(
            x, trends['sc_trend'], 
            title='State Coverage Growth', 
            ylabel='Cumulative Unique Inputs', 
            filename=PLOT_SC_CURVE,
            color='#3498db' 
        )
    
    # --- 调用新绘图函数 ---
    plot_crash_seed_distribution(log_data, PLOT_SEED_DIST)

    print("\nAll curves generated successfully.")

if __name__ == "__main__":
    main()