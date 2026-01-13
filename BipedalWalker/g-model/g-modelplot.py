import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE
import time
from collections import Counter # 新增引入

# ==========================================
# 全局配置与风格设置
# ==========================================
sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)

# 颜色配置
COLORS = {
    'bar_rand': '#95a5a6',    # 柱状图-Random (灰色)
    'bar_gen': '#e67e22',     # 柱状图-Generative (橙色)
    'line_total': '#d35400',  # 时间图-总数 (深橙色)
    'tsne_safe': '#3498db',   # t-SNE-安全点 (蓝色)
    'tsne_crash': '#e74c3c',  # t-SNE-崩溃点 (红色)
}

# 路径配置
# 注意：请根据 test_gen.py 实际生成的文件夹名称修改此处
RESULT_DIR = os.path.join("results", "generative+novelty_50_seed_0")
LOG_FILENAME = os.path.join(RESULT_DIR, "all_test_cases_log.pkl")
PLOT_5_FILE = os.path.join(RESULT_DIR, '5_behaviour_coverage_heatmap.png') # 新增

def load_data():
    """加载日志数据"""
    if not os.path.exists(LOG_FILENAME):
        print(f"错误：未找到日志文件 {LOG_FILENAME}")
        print(f"请检查 result 文件夹路径是否正确: {RESULT_DIR}")
        return None
    
    print(f"正在加载日志: {LOG_FILENAME} ...")
    with open(LOG_FILENAME, 'rb') as f:
        log_data = pickle.load(f)
    return log_data

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
    """处理数据并绘制两张柱状图 (Raw & Unique)"""
    
    # --- 1. 去重前 (Raw Data) ---
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

    # --- 2. 去重后 (Unique Data) ---
    unique_data = []
    seen_inputs = set()
    
    for entry in log_data:
        # 将 input 列表转为 tuple 以便用于 set 去重
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
    """绘制所有唯一 Crash 随时间变化的累计图（横坐标：小时）"""
    if not log_data:
        return

    # 提取所有唯一 Crash 的时间戳
    unique_crashes_timestamps = {} 
    
    for entry in log_data:
        if entry.get('is_crash'):
            t_in = tuple(entry['input'])
            timestamp = entry.get('time', 0)
            
            if t_in not in unique_crashes_timestamps:
                unique_crashes_timestamps[t_in] = timestamp

    # 按时间排序
    sorted_times = sorted(list(unique_crashes_timestamps.values()))
    
    if not sorted_times:
        print("未发现 Unique Crash，跳过时间图绘制。")
        return

    # 将秒转换为小时
    times_in_hours = [t / 3600.0 for t in sorted_times]
    counts = np.arange(1, len(times_in_hours) + 1)

    # 绘图
    plt.figure(figsize=(10, 6))
    
    plt.plot(times_in_hours, counts, color=COLORS['line_total'], linewidth=3, label='Total Unique Crashes')
    plt.fill_between(times_in_hours, counts, color=COLORS['line_total'], alpha=0.1)

    plt.title('Cumulative Unique Crashes Over Time', fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Time (Hours)', fontsize=14)
    plt.ylabel('Cumulative Count', fontsize=14)
    
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    stats_text = (
        f"$\\bf{{Summary}}$\n"
        f"Total Unique Crashes: {len(counts)}\n"
        f"Final Time: {times_in_hours[-1]:.2f} h"
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

# ==========================================
# t-SNE 相关功能
# ==========================================

def run_tsne(data, n_samples):
    """
    运行 t-SNE 降维
    """
    if n_samples < 50:
        print(f"数据点较少 ({n_samples}个)，自动调整 Perplexity。")
        perplexity_value = max(5, n_samples - 1)
    else:
        perplexity_value = min(30, n_samples - 1)
        
    print(f"正在对 {n_samples} 个唯一输入运行 t-SNE (Perplexity={perplexity_value})...")
    
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

def plot_tsne_distribution(log_data):
    """
    提取所有唯一输入，并绘制 t-SNE 散点图
    红色 = Crash, 蓝色 = Safe
    """
    print("\n=== 开始绘制 t-SNE 分布图 ===")
    
    unique_inputs = []
    labels = []  # 0: Safe, 1: Crash
    seen_inputs = set()

    # 1. 数据准备与去重
    for entry in log_data:
        t_in = tuple(entry['input'])
        if t_in not in seen_inputs:
            seen_inputs.add(t_in)
            
            # 将 tuple 转回 numpy array
            input_arr = np.array(t_in)
            unique_inputs.append(input_arr)
            
            # 标记 Label
            if entry.get('is_crash'):
                labels.append(1)
            else:
                labels.append(0)
    
    if not unique_inputs:
        print("未找到有效输入数据，跳过 t-SNE。")
        return

    X = np.array(unique_inputs)
    y = np.array(labels)
    n_samples = X.shape[0]

    # 2. 运行 t-SNE
    tsne_results = run_tsne(X, n_samples)
    
    # 3. 绘图
    plt.figure(figsize=(12, 10))
    
    # 分离数据点
    safe_mask = (y == 0)
    crash_mask = (y == 1)
    
    safe_points = tsne_results[safe_mask]
    crash_points = tsne_results[crash_mask]
    
    # 绘制安全点 (蓝色, 半透明)
    plt.scatter(
        safe_points[:, 0], safe_points[:, 1], 
        c=COLORS['tsne_safe'], alpha=0.3, s=20, 
        label=f'Safe Inputs ({len(safe_points)})'
    )
    
    # 绘制崩溃点 (红色, 显眼)
    if len(crash_points) > 0:
        plt.scatter(
            crash_points[:, 0], crash_points[:, 1], 
            c=COLORS['tsne_crash'], alpha=0.9, s=30, marker='X',
            label=f'Crashes ({len(crash_points)})'
        )
    
    plt.title('t-SNE Visualization of Input Space\n(Red=Crash, Blue=Safe)', fontweight='bold', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    sns.despine()
    plt.tight_layout()
    
    filename = '4_input_space_tsne.png'
    save_path = os.path.join(RESULT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存: {filename}")
    plt.close()

# ==========================================
# [新增] Behaviour Diversity / Coverage 相关功能
# ==========================================
def calculate_behaviour_diversity(log_data, grid_size=(50, 50)):
    """
    [新增] 计算基于 2D 网格 (Distance, Hull Angle) 的行为多样性 (覆盖率)。
    这对应于 Behaviour Diversity 和 Fault Diversity 指标。
    """
    print(f"\n{'='*40}\n       Behaviour Diversity Analysis\n{'='*40}")
    
    # 1. 提取行为特征 (BDs)
    dists = []
    angles = []
    is_crash_list = []
    
    found_bd = False
    
    for entry in log_data:
        # 使用 .get 以兼容旧日志文件
        d = entry.get('bd_distance')
        a = entry.get('bd_mean_angle')
        c = entry.get('is_crash', False)
        
        if d is not None and a is not None:
            dists.append(d)
            angles.append(a)
            is_crash_list.append(c)
            found_bd = True
            
    if not found_bd:
        print("Warning: No behavior descriptors ('bd_distance', 'bd_mean_angle') found in log.")
        print("Please rerun test_gen.py with the modified code to generate these metrics.")
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
    
    # 构建热力图矩阵
    heatmap = np.zeros(grid_size)

    for i in range(len(dists)):
        bin_id = (dist_indices[i], angle_indices[i])
        filled_bins.add(bin_id)
        heatmap[dist_indices[i], angle_indices[i]] += 1
        
        if is_crash_list[i]:
            filled_crash_bins.add(bin_id)
            
    # 4. 输出报告
    total_bins = grid_size[0] * grid_size[1]
    print(f"  Behaviour Coverage (State Coverage):    {len(filled_bins)} / {total_bins} ({len(filled_bins)/total_bins:.2%})")
    print(f"  Fault Diversity (Unique Crash Types):   {len(filled_crash_bins)}")
    print(f"{'='*40}\n")
    
    # 5. 绘制覆盖率热力图
    plt.figure(figsize=(10, 8))
    # 使用 log1p (log(1+x)) 使得低频和高频格子都能看清
    plt.imshow(np.log1p(heatmap).T, origin='lower', aspect='auto', cmap='viridis', 
               extent=[min_dist, max_dist, min_angle, max_angle])
    plt.colorbar(label='Log(Count)')
    plt.title(f'Behaviour Space Coverage (Filled Bins: {len(filled_bins)})')
    plt.xlabel('Distance')
    plt.ylabel('Mean Hull Angle')
    
    plt.savefig(PLOT_5_FILE, dpi=300)
    print(f"图表已保存: {PLOT_5_FILE}")
    plt.close()

def main():
    # 检查结果目录是否存在
    if not os.path.exists(RESULT_DIR):
        print(f"错误：目录 {RESULT_DIR} 不存在。")
        print("请运行 test_gen.py 生成数据，或修改本脚本中的 RESULT_DIR 变量。")
        return

    log_data = load_data()
    
    if log_data:
        print("\n=== 开始绘制图表 ===")
        
        # 1. 绘制柱状图 (Raw & Unique)
        process_and_plot_bars(log_data)
        
        # 2. 绘制时间曲线图 (Total Only - Time in Hours)
        plot_total_crashes_over_time(log_data)
        
        # 3. 绘制 t-SNE 输入空间分布图
        plot_tsne_distribution(log_data)

        # 4. [新增] 绘制行为多样性热力图与计算覆盖率
        calculate_behaviour_diversity(log_data)
        
    print("\n=== 所有绘图任务完成 ===")

if __name__ == "__main__":
    main()