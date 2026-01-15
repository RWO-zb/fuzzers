import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ==========================================
# 全局配置
# ==========================================
sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)

# 颜色配置
COLORS = {
    'bar_rand': '#95a5a6',    # 柱状图-Random (灰色)
    'bar_gen': '#e67e22',     # 柱状图-Generative (橙色)
    'line_total': '#d35400',  # 时间图-总数 (深橙色)
    'traj_crash': '#c0392b',  # 轨迹图-Crash (深红色)
    'traj_normal': '#bdc3c7'  # 轨迹图-正常 (浅灰色)
}

# 文件路径配置 (请根据实际情况修改 RESULT_DIR)
RESULT_DIR = "results/MC_generative+novelty_50_seed_1022"
LOG_FILENAME = os.path.join(RESULT_DIR, "all_test_cases_log.pkl")
TRAJ_FILENAME = os.path.join(RESULT_DIR, "all_trajectories.pkl")

def load_data():
    """加载日志和轨迹数据"""
    if not os.path.exists(LOG_FILENAME):
        print(f"错误：未找到日志文件 {LOG_FILENAME}")
        return None, None
    
    print(f"正在加载日志: {LOG_FILENAME} ...")
    with open(LOG_FILENAME, 'rb') as f:
        log_data = pickle.load(f)
        
    traj_data = None
    if os.path.exists(TRAJ_FILENAME):
        print(f"正在加载轨迹: {TRAJ_FILENAME} ...")
        with open(TRAJ_FILENAME, 'rb') as f:
            traj_data = pickle.load(f)
    else:
        print("警告：未找到轨迹文件，将跳过轨迹图绘制。")
        
    return log_data, traj_data

def plot_bar_chart(rand_count, gen_count, title_text, filename, total_samples):
    """通用柱状图绘制函数"""
    plt.figure(figsize=(8, 6))
    
    labels = ['Random', 'Generative']
    counts = [rand_count, gen_count]
    colors = [COLORS['bar_rand'], COLORS['bar_gen']]
    
    bars = plt.bar(labels, counts, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    
    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 标题和标签
    plt.title(f"{title_text}\n(Total Samples: {total_samples})", fontweight='bold', fontsize=16, pad=15)
    plt.ylabel('Crash Count', fontsize=14)
    plt.grid(axis='x') # 隐藏横向网格，只留纵向
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename), dpi=300)
    print(f"图表已保存: {filename}")
    plt.close()

def process_and_plot_bars(log_data):
    """处理数据并绘制两张柱状图"""
    # --- 1. 去重前 (Raw) ---
    raw_crashes = [e for e in log_data if e.get('is_crash')]
    raw_rand = sum(1 for c in raw_crashes if c.get('source') == 'random')
    raw_gen = sum(1 for c in raw_crashes if c.get('source') == 'generative')
    
    plot_bar_chart(
        raw_rand, 
        raw_gen, 
        'Total Crashes Detected (Raw / Before Dedup)', 
        '1_crash_counts_raw.png',
        len(log_data)
    )

    # --- 2. 去重后 (Unique) ---
    unique_data = []
    seen_inputs = set()
    for entry in log_data:
        t_in = tuple(entry['input'])
        if t_in not in seen_inputs:
            seen_inputs.add(t_in)
            unique_data.append(entry)
            
    unique_crashes = [e for e in unique_data if e.get('is_crash')]
    uniq_rand = sum(1 for c in unique_crashes if c.get('source') == 'random')
    uniq_gen = sum(1 for c in unique_crashes if c.get('source') == 'generative')
    
    plot_bar_chart(
        uniq_rand, 
        uniq_gen, 
        'Unique Crashes Detected (After Dedup)', 
        '2_crash_counts_unique.png',
        len(unique_data)
    )

def plot_total_crashes_over_time(log_data):
    """绘制所有唯一 Crash 随时间变化的累计图（合并来源）"""
    if not log_data:
        return

    # 提取所有唯一 Crash 的时间戳
    unique_crashes_timestamps = {} # Key: input_tuple, Value: first_timestamp
    
    for entry in log_data:
        if entry.get('is_crash'):
            t_in = tuple(entry['input'])
            timestamp = entry.get('timestamp', 0)
            
            # 如果是新发现的 unique crash，记录时间
            if t_in not in unique_crashes_timestamps:
                unique_crashes_timestamps[t_in] = timestamp

    # 排序时间戳
    sorted_times = sorted(list(unique_crashes_timestamps.values()))
    
    if not sorted_times:
        print("未发现 Crash，跳过时间图绘制。")
        return

    # 转换为分钟
    times_in_min = [t / 60.0 for t in sorted_times]
    # 累计计数 (1, 2, 3 ...)
    counts = np.arange(1, len(times_in_min) + 1)

    # 绘图
    plt.figure(figsize=(10, 6))
    
    plt.plot(times_in_min, counts, color=COLORS['line_total'], linewidth=3, label='Total Unique Crashes')
    plt.fill_between(times_in_min, counts, color=COLORS['line_total'], alpha=0.1)

    plt.title('Cumulative Unique Crashes Over Time', fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Time (Minutes)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # 强制 Y 轴为整数
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 添加统计框
    stats_text = (
        f"$\\bf{{Summary}}$\n"
        f"Total Unique Crashes: {len(counts)}\n"
        f"Time Elapsed: {times_in_min[-1]:.1f} min"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.95, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.grid(True, linestyle='--', alpha=0.6)
    sns.despine()
    plt.tight_layout()
    
    filename = '3_crash_time_series_total.png'
    plt.savefig(os.path.join(RESULT_DIR, filename), dpi=300)
    print(f"图表已保存: {filename}")
    plt.close()

def plot_total_trajectory(log_data, traj_data):
    """绘制状态空间轨迹图（Crash 不区分来源，统一显示）"""
    if not log_data or not traj_data:
        return
    
    if len(log_data) != len(traj_data):
        print("警告: 日志与轨迹数据长度不一致。")
        return

    plt.figure(figsize=(10, 8))

    # MountainCar 边界线
    plt.axvline(x=-1.2, color='black', linestyle='--', alpha=0.4, linewidth=1)
    plt.axvline(x=0.6, color='black', linestyle='--', alpha=0.4, linewidth=1)

    # 数据容器
    normal_points = []
    crash_points = []

    # 采样率（避免正常轨迹点过多导致渲染慢）
    NORMAL_SAMPLE_RATE = 5 

    for i, entry in enumerate(log_data):
        is_crash = entry.get('is_crash')
        traj = np.array(traj_data[i])
        
        if is_crash:
            # Crash 轨迹：全部保留
            crash_points.append(traj)
        else:
            # 正常轨迹：降采样
            if i % NORMAL_SAMPLE_RATE == 0:
                normal_points.append(traj)

    # 1. 绘制正常轨迹 (背景)
    if normal_points:
        all_normal = np.vstack(normal_points)
        # 如果点非常多，进一步随机采样
        if len(all_normal) > 100000:
            idx = np.random.choice(len(all_normal), 100000, replace=False)
            all_normal = all_normal[idx]
            
        plt.scatter(all_normal[:, 0], all_normal[:, 1], c=COLORS['traj_normal'], s=5, alpha=0.15, 
                    label='Normal Trajectories', edgecolors='none', rasterized=True)

    # 2. 绘制所有 Crash 轨迹 (前景，统一颜色)
    if crash_points:
        all_crash = np.vstack(crash_points)
        plt.scatter(all_crash[:, 0], all_crash[:, 1], c=COLORS['traj_crash'], s=15, alpha=0.6, 
                    label='Crash Trajectories', marker='x')

    plt.title('State Space Coverage (All Crashes)', fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Velocity', fontsize=14)
    
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=12)
    
    # 统计信息
    stats_text = (
        f"$\\bf{{Stats}}$\n"
        f"Normal Trajs (sampled): {len(normal_points)}\n"
        f"Total Crashes: {len(crash_points)}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='left', bbox=props)

    sns.despine()
    plt.tight_layout()
    
    filename = '4_state_space_trajectories_total.png'
    plt.savefig(os.path.join(RESULT_DIR, filename), dpi=300)
    print(f"图表已保存: {filename}")
    plt.close()

def main():
    # 确保输出目录存在
    if not os.path.exists(RESULT_DIR):
        print(f"错误：目录 {RESULT_DIR} 不存在，请修改脚本中的 RESULT_DIR 变量。")
        return

    log_data, traj_data = load_data()
    
    if log_data:
        print("\n=== 开始绘制图表 ===")
        # 绘制图1和图2 (柱状图)
        process_and_plot_bars(log_data)
        
        # 绘制图3 (时间图)
        plot_total_crashes_over_time(log_data)
        
    if log_data and traj_data:
        # 绘制图4 (轨迹图)
        plot_total_trajectory(log_data, traj_data)
        
    print("\n=== 所有任务完成 ===")

if __name__ == "__main__":
    main()