import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns
from matplotlib.patches import Rectangle

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] # 尝试支持中文，如果乱码会回退
plt.rcParams['axes.unicode_minus'] = False

def load_latest_trace_data(base_path='./results'):
    """自动查找最新的结果文件夹并加载 trace_data.pkl"""
    if not os.path.exists(base_path):
        print(f"错误：找不到路径 {base_path}")
        return None, None

    folders = glob.glob(os.path.join(base_path, '*'))
    if not folders:
        print("未找到任何结果文件夹。")
        return None, None
    
    # 按修改时间排序，找最新的
    latest_folder = max(folders, key=os.path.getmtime)
    trace_file = os.path.join(latest_folder, 'trace_data.pkl')
    
    print(f"--> 锁定最新实验文件夹: {latest_folder}")
    print(f"--> 正在加载数据文件: trace_data.pkl ...")
    
    if not os.path.exists(trace_file):
        print("错误：该文件夹下没有 trace_data.pkl。请先运行修改后的 enjoy_cure.py。")
        return None, None
        
    with open(trace_file, 'rb') as f:
        return pickle.load(f), latest_folder

def plot_single_attribution(trace, save_path, episode_idx):
    """为单个 Crash Episode 绘制详细归因图并保存"""
    steps = trace["steps"]
    positions = trace["positions"]
    gaps = trace["action_gaps"]
    actions = trace["actions"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- 图 1: 轨迹 + 关键度热力图 ---
    ax1.plot(steps, positions, color='#d62728', linewidth=2, label='Crash Trajectory')
    ax1.axhline(0.5, color='green', linestyle='--', label='Goal (0.5)')
    ax1.axhline(-1.2, color='gray', linestyle='--', alpha=0.5)
    
    # 核心：用红色深浅表示 Action Gap 大小
    max_gap = max(gaps) if gaps else 1.0
    
    # 绘制高亮背景
    for i in range(len(steps) - 1):
        importance = gaps[i] / max_gap
        if importance > 0.15: # 阈值过滤
            ax1.axvspan(steps[i], steps[i+1], color='red', alpha=importance * 0.5, lw=0)
            
    ax1.set_ylabel('Position')
    ax1.set_title(f'Episode {episode_idx} (Crash) - Critical Steps Attribution', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # --- 图 2: Action Gap 曲线 + 动作 ---
    ax2.plot(steps, gaps, color='purple', linewidth=1.5, label='Action Gap (Confidence)')
    
    # 标记动作
    threshold = max_gap * 0.2
    for s, g, a in zip(steps, gaps, actions):
        if g > threshold:
            marker = '<' if a == 0 else ('>' if a == 2 else 'o')
            color = 'orange' if a == 0 else ('red' if a == 2 else 'gray')
            if a != 1: # 忽略不动作的点，避免太乱
                ax2.scatter(s, g, marker=marker, color=color, s=40, zorder=5)
    
    ax2.set_ylabel('Gap Value')
    ax2.set_xlabel('Time Step')
    ax2.set_title('Decision Confidence (Gap)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig) # 关闭图像释放内存

def analyze_all_crashes(traces, output_folder):
    """主分析逻辑"""
    
    # 1. 数据分类
    crash_traces = [t for t in traces if t.get('did_crash', False)]
    normal_traces = [t for t in traces if not t.get('did_crash', False)]
    
    n_crash = len(crash_traces)
    n_total = len(traces)
    
    print(f"\n{'='*30}")
    print(f" 数据概览 (Data Overview)")
    print(f"{'='*30}")
    print(f"Total Episodes : {n_total}")
    print(f"Crash Episodes : {n_crash}")
    print(f"Crash Rate     : {n_crash/n_total*100:.2f}%")
    
    if n_crash == 0:
        print("未检测到 Crash，分析结束。")
        return

    # 创建保存图片的子文件夹
    plots_dir = os.path.join(output_folder, 'crash_analysis_plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\n--> 所有分析图表将保存至: {plots_dir}")

    # =========================================================
    # 分析 1: Crash 轨迹相位图聚类 (Phase Portrait)
    # =========================================================
    print("正在生成：所有 Crash 的相位图聚类...")
    plt.figure(figsize=(10, 6))
    
    # 先画一些正常的轨迹做背景（淡灰色）
    for t in normal_traces[:50]: # 最多画50条避免卡顿
        plt.plot(t['positions'], t['velocities'], color='gray', alpha=0.1, linewidth=0.5)
        
    # 再画 Crash 轨迹（红色）
    for t in crash_traces:
        plt.plot(t['positions'], t['velocities'], color='red', alpha=0.3, linewidth=1)
        # 标记终点
        plt.scatter(t['positions'][-1], t['velocities'][-1], color='darkred', s=10, marker='x')

    plt.title(f"Phase Space: Crash (Red) vs Normal (Gray)\nTotal Crashes: {n_crash}", fontsize=14)
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.xlim(-1.2, 0.6)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "1_all_crashes_phase_portrait.png"), dpi=300)
    plt.close()

    # =========================================================
    # 分析 2: Action Gap 统计对比 (Confidence Check)
    # =========================================================
    print("正在生成：Action Gap 统计对比箱线图...")
    
    # 提取平均 Gap 和 最大 Gap
    crash_avg_gaps = [np.mean(t['action_gaps']) for t in crash_traces]
    normal_avg_gaps = [np.mean(t['action_gaps']) for t in normal_traces]
    
    crash_max_gaps = [np.max(t['action_gaps']) for t in crash_traces]
    normal_max_gaps = [np.max(t['action_gaps']) for t in normal_traces]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Boxplot 1: Average Gap
    data_avg = [normal_avg_gaps, crash_avg_gaps]
    axes[0].boxplot(data_avg, labels=['Normal', 'Crash'], patch_artist=True, 
                    boxprops=dict(facecolor="lightblue"))
    axes[0].set_title("Average Action Gap (Confidence)")
    axes[0].set_ylabel("Mean Q-Gap")
    
    # Boxplot 2: Max Gap
    data_max = [normal_max_gaps, crash_max_gaps]
    axes[1].boxplot(data_max, labels=['Normal', 'Crash'], patch_artist=True,
                    boxprops=dict(facecolor="salmon"))
    axes[1].set_title("Max Action Gap (Peak Confidence)")
    axes[1].set_ylabel("Max Q-Gap")
    
    plt.suptitle("Does the agent feel more 'confused' (low gap) during crashes?", fontsize=14)
    plt.savefig(os.path.join(plots_dir, "2_gap_statistics_comparison.png"), dpi=300)
    plt.close()

    # =========================================================
    # 分析 3: 逐个生成详细归因图 (Individual Trace Analysis)
    # =========================================================
    print(f"正在为所有 {n_crash} 个 Crash 生成详细归因图...")
    
    # 限制最大生成数量，防止如果有1000个crash把硬盘写满
    max_plots = 50 
    
    for i, trace in enumerate(crash_traces):
        if i >= max_plots:
            print(f"  (已达到最大绘图限制 {max_plots}，剩余略过)")
            break
            
        save_name = os.path.join(plots_dir, f"crash_trace_{i:03d}.png")
        plot_single_attribution(trace, save_name, i)
        
    print(f"详细归因图生成完毕！请查看文件夹: {plots_dir}")

if __name__ == "__main__":
    data, folder_path = load_latest_trace_data()
    if data:
        analyze_all_crashes(data, folder_path)