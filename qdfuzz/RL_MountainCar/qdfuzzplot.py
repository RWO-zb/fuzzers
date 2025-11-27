import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import ast
from matplotlib.ticker import MaxNLocator

# --- 1. 配置 (请修改为您实际的文件路径) ---
CSV_FILE = 'results/mc_test_data.csv'  
OBS_FILE = 'results/mc_test_obs.txt'

# 输出图片名称
PLOT_1_NAME = '1_crash_discovery_over_time.png'
PLOT_2_NAME = '2_state_space_trajectory.png'
PLOT_3_NAME = '3_mutation_depth_hist.png'

# --- 2. 全局风格设置 (严格保持 curefuzzplot.py 风格) ---
sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)

def load_data(csv_path, obs_path):
    """
    加载 CSV 数据用于 Plot 1 & 3
    加载 OBS 轨迹数据用于 Plot 2 (基于 OBS header 中的 Oracle 判断 Crash)
    """
    print(f"正在加载数据...\nCSV: {csv_path}\nOBS: {obs_path}")
    
    # --- A. 读取 CSV (主要用于 Plot 1: 时间趋势 和 Plot 3: 变异代数) ---
    selection_log = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # 解析 input 用于去重 (如果需要)
                inp = row['input']
                if isinstance(inp, str):
                    try:
                        inp = ast.literal_eval(inp)
                    except:
                        pass
                
                entry = {
                    'did_crash': bool(row['is_faulty']),
                    'crash_time': row['discovery_time'],
                    # 适配 Plot 3 逻辑: 原始代码用 parent_depth + 1，这里 mutation_count 已经是代数了
                    # 为了保持兼容，我们存为 parent_depth = count - 1
                    'parent_depth': int(row['mutation_count']) - 1, 
                    'mutate_state': np.array(inp, dtype=np.float32) if isinstance(inp, list) else inp
                }
                selection_log.append(entry)
        except Exception as e:
            print(f"读取 CSV 出错: {e}")
    else:
        print(f"警告: 未找到 CSV 文件 {csv_path}")

    # --- B. 读取 OBS 文件 (主要用于 Plot 2: 轨迹) ---
    # 结构: [ {'trajectory': [[x,y],...], 'is_crash': bool}, ... ]
    obs_data = []
    
    if os.path.exists(obs_path):
        current_seq = []
        current_is_crash = False # 默认为 False
        has_header = False
        
        with open(obs_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if line.startswith('--- Test Case Info:'):
                    # 1. 保存上一条轨迹
                    if has_header and current_seq: 
                        obs_data.append({
                            'trajectory': np.array(current_seq),
                            'is_crash': current_is_crash
                        })
                    
                    # 2. 重置并解析新的 Header
                    current_seq = []
                    has_header = True
                    try:
                        # 提取 JSON 部分: "--- Test Case Info: { ... } ---"
                        json_str = line.split('--- Test Case Info:')[1].rsplit('---', 1)[0].strip()
                        info = json.loads(json_str)
                        
                        # [关键修改] 根据 OBS 文件里的 Oracle 字段判断
                        # Oracle: true -> Crash, Oracle: false -> Normal
                        current_is_crash = bool(info.get('Oracle', False))
                        
                    except Exception as e:
                        print(f"解析 OBS Header 出错: {e}")
                        current_is_crash = False # 默认 fallback
                        
                else:
                    # 读取坐标点
                    try:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            current_seq.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass
            
            # 保存最后一条
            if has_header and current_seq:
                obs_data.append({
                    'trajectory': np.array(current_seq),
                    'is_crash': current_is_crash
                })
    else:
        print(f"警告: 未找到 OBS 文件 {obs_path}")

    return selection_log, obs_data

def deduplicate_log(selection_log):
    """仅对 Log 数据进行去重 (用于 Plot 1 & 3)"""
    seen_states = set()
    dedup_log = []
    
    for entry in selection_log:
        state = entry.get('mutate_state')
        if state is None: continue
        
        state_bytes = state.tobytes()
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            dedup_log.append(entry)
    return dedup_log

def plot_1_crashes_over_time(selection_log, total_samples_count):
    dedup_samples_count = len(selection_log)
    crash_times = [e.get('crash_time') for e in selection_log if e.get('did_crash', False) and e.get('crash_time') is not None]
    unique_crashes_count = len(crash_times)
    
    if not crash_times: 
        print("Plot 1: 无 Crash 数据。")
        return

    crash_times.sort()
    times_in_hours = [t / 3600.0 for t in crash_times]
    counts = range(1, len(crash_times) + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(times_in_hours, counts, color='#E64A19', linewidth=3, label='Unique Crashes')
    plt.fill_between(times_in_hours, counts, color='#E64A19', alpha=0.1)
    
    plt.title('Crash Discovery Over Time', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Time (Hours)', fontsize=14, labelpad=10)
    plt.ylabel('Cumulative Crashes', fontsize=14, labelpad=10)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.grid(True, linestyle='--', alpha=0.6)
    
    stats_text = (
        f"$\\bf{{Statistics}}$\n"
        f"Total Samples: {total_samples_count}\n"
        f"Dedup. Samples: {dedup_samples_count}\n"
        f"Unique Crashes: {unique_crashes_count}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=13,
                   verticalalignment='top', horizontalalignment='left', bbox=props)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_1_NAME, dpi=300)
    plt.close()
    print(f"Plot 1 Saved: {PLOT_1_NAME}")

def plot_2_state_space(obs_data):
    """
    完全基于 OBS 文件中的 oracle 字段绘制
    """
    if not obs_data:
        print("Plot 2: 无轨迹数据。")
        return

    normal_points, crash_points = [], []

    # 遍历所有 episode
    for item in obs_data:
        seq_arr = item['trajectory']
        is_crash = item['is_crash'] # 直接使用 OBS header 里的 Oracle 判断
        
        if len(seq_arr) == 0: continue

        if is_crash:
            crash_points.append(seq_arr)
        else:
            normal_points.append(seq_arr)

    plt.figure(figsize=(12, 8))
    
    # 绘制 Normal (灰色 #B0BEC5)
    if normal_points:
        all_normal = np.vstack(normal_points)
        # 降采样优化
        if len(all_normal) > 100000:
            indices = np.random.choice(len(all_normal), 100000, replace=False)
            all_normal = all_normal[indices]
        plt.scatter(all_normal[:, 0], all_normal[:, 1], c='#B0BEC5', s=10, alpha=0.3, label='Normal Episodes', edgecolors='none')
    
    # 绘制 Crash (红色 #D32F2F)
    if crash_points:
        all_crash = np.vstack(crash_points)
        plt.scatter(all_crash[:, 0], all_crash[:, 1], c='#D32F2F', s=20, alpha=0.8, label='Crash Episodes', marker='x')

    plt.title('State Space Coverage: Normal vs. Crash Episodes', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Position', fontsize=14, labelpad=10)
    plt.ylabel('Velocity', fontsize=14, labelpad=10)
    
    # MountainCar 边界
    plt.axvline(x=-1.2, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0.6, color='k', linestyle='--', alpha=0.3)
    
    plt.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=12)
    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_2_NAME, dpi=300)
    plt.close()
    print(f"Plot 2 Saved: {PLOT_2_NAME}")

def plot_3_mutation_depth(selection_log):
    crash_depths = []
    for entry in selection_log:
        if entry.get('did_crash', False):
            depth = entry.get('parent_depth')
            crash_depths.append(0 if depth is None else depth + 1)
            
    if not crash_depths: 
        print("Plot 3: 无 Crash 数据。")
        return

    mean_gen = np.mean(crash_depths)
    median_gen = np.median(crash_depths)
    max_gen = np.max(crash_depths)
    
    plt.figure(figsize=(12, 7))
    max_x = int(max_gen)
    bins = np.arange(0, max_x + 2) - 0.5 

    n, bins, patches = plt.hist(crash_depths, bins=bins, color='#009688', edgecolor='white', alpha=0.85, rwidth=0.8)
    
    plt.title('Distribution of Crashes by Mutation Generation', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Generation (Depth)', fontsize=14, labelpad=10)
    plt.ylabel('Count of Crashes', fontsize=14, labelpad=10)
    
    if max_x < 20:
        plt.xticks(range(0, max_x + 1))
    else:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    for i in range(len(patches)):
        if n[i] > 0:
            plt.text(patches[i].get_x() + patches[i].get_width()/2, n[i], int(n[i]), ha='center', va='bottom', fontsize=11, fontweight='bold', color='#455A64')

    stats_text = (
        f"$\\bf{{Statistics}}$\n"
        f"Mean: {mean_gen:.2f}\n"
        f"Median: {median_gen:.1f}\n"
        f"Max: {int(max_gen)}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=13, verticalalignment='top', horizontalalignment='right', bbox=props)

    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_3_NAME, dpi=300)
    plt.close()
    print(f"Plot 3 Saved: {PLOT_3_NAME}")

def main():
    # 1. 加载数据
    raw_selection_log, obs_data = load_data(CSV_FILE, OBS_FILE)
    
    if raw_selection_log:
        # 去重 CSV 数据用于 Plot 1 & 3
        dedup_log = deduplicate_log(raw_selection_log)
        
        plot_1_crashes_over_time(dedup_log, len(raw_selection_log))
        plot_3_mutation_depth(dedup_log)
    
    if obs_data:
        # 使用 OBS 数据绘制 Plot 2 (基于 Header 中的 Oracle)
        plot_2_state_space(obs_data)

if __name__ == "__main__":
    main()