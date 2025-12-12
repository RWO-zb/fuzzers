import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ==========================================
# 全局配置与风格设置
# ==========================================
sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)

# 颜色配置
COLORS = {
    'bar_rand': '#95a5a6',    # 柱状图-Random (灰色)
    'bar_gen': '#e67e22',     # 柱状图-Generative (橙色)
    'line_total': '#d35400',  # 时间图-总数 (深橙色)
}

# 路径配置
# 注意：请根据 test_gen.py 实际生成的文件夹名称修改此处
# test_gen.py 默认参数生成的文件夹通常是 "generative_50_seed_0"
RESULT_DIR = os.path.join("results", "generative+novelty_50_seed_1022")
LOG_FILENAME = os.path.join(RESULT_DIR, "all_test_cases_log.pkl")

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
    """绘制所有唯一 Crash 随时间变化的累计图（合并来源）"""
    if not log_data:
        return

    # 提取所有唯一 Crash 的时间戳
    # Key: input_tuple, Value: first_timestamp (time)
    unique_crashes_timestamps = {} 
    
    for entry in log_data:
        if entry.get('is_crash'):
            t_in = tuple(entry['input'])
            # BipedalWalker 代码中使用的是 "time" 字段表示 elapsed time
            timestamp = entry.get('time', 0)
            
            # 如果是新发现的 unique crash，记录发现时间
            if t_in not in unique_crashes_timestamps:
                unique_crashes_timestamps[t_in] = timestamp

    # 按时间排序
    sorted_times = sorted(list(unique_crashes_timestamps.values()))
    
    if not sorted_times:
        print("未发现 Unique Crash，跳过时间图绘制。")
        return

    # 将秒转换为分钟
    times_in_min = [t / 60.0 for t in sorted_times]
    # 累计计数 (1, 2, 3 ...)
    counts = np.arange(1, len(times_in_min) + 1)

    # 绘图
    plt.figure(figsize=(10, 6))
    
    plt.plot(times_in_min, counts, color=COLORS['line_total'], linewidth=3, label='Total Unique Crashes')
    plt.fill_between(times_in_min, counts, color=COLORS['line_total'], alpha=0.1)

    plt.title('Cumulative Unique Crashes Over Time', fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Time (Minutes)', fontsize=14)
    plt.ylabel('Cumulative Count', fontsize=14)
    
    # 强制 Y 轴刻度为整数
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 添加统计信息框
    stats_text = (
        f"$\\bf{{Summary}}$\n"
        f"Total Unique Crashes: {len(counts)}\n"
        f"Final Time: {times_in_min[-1]:.1f} min"
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
        
        # 2. 绘制时间曲线图 (Total Only)
        plot_total_crashes_over_time(log_data)
        
    print("\n=== 所有绘图任务完成 ===")

if __name__ == "__main__":
    main()