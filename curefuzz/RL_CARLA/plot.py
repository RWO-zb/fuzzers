import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ==========================================
# 1. 全局绘图风格设置 (Academic Style)
# ==========================================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'  # 来自 plot.py 的数学字体设置
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.5

# ==========================================
# 2. 绘图功能函数
# ==========================================

def plot_crash_curve(df, output_dir="."):
    """
    来自 plot.py: 绘制随时间变化的累计碰撞次数 (Cumulative Crashes)
    """
    print("--- Plotting Crash Curve ---")
    
    # 检查必要列
    if 'elapsed_time' not in df.columns or 'collision' not in df.columns or 'phase' not in df.columns:
        print("[Skip] Missing columns for crash curve.")
        return

    fuzz_df = df[df['phase'] == 'Phase2'].sort_values(by='elapsed_time')
    
    if len(fuzz_df) == 0:
        print("[Info] No Phase2 data for crash curve. Skipping.")
        return

    crashes = fuzz_df[fuzz_df['collision'] == True].copy()
    crashes['cumulative_crashes'] = range(1, len(crashes) + 1)
    crashes['elapsed_hours'] = crashes['elapsed_time'] / 3600.0

    plt.figure(figsize=(6, 4))

    # 绘制阶梯图
    plt.step(crashes['elapsed_hours'], crashes['cumulative_crashes'], 
             where='post', linewidth=2, color='#00529F', label='CURE')

    # 绘制散点标记
    plt.scatter(crashes['elapsed_hours'], crashes['cumulative_crashes'], 
                color='#00529F', s=25, marker='o', edgecolors='white', zorder=5)

    plt.xlabel('Time (h)', fontsize=14)
    plt.ylabel('# of Unique Crashes', fontsize=14)

    # 美化边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.grid(True, linestyle=':', alpha=0.5, axis='y') 
    plt.legend(frameon=False, fontsize=12, loc='lower right')
    plt.tight_layout()

    save_path = os.path.join(output_dir, "crash_curve_academic.png")
    plt.savefig(save_path, dpi=300)
    # plt.savefig(os.path.join(output_dir, "crash_curve_academic.pdf"), format='pdf') # 可选 PDF
    print(f"[Success] Saved: {save_path}")
    plt.close() # 关闭当前图表以释放内存


def plot_behavior_diversity(df, output_dir="."):
    """
    来自 plot2.py: 绘制行为覆盖率和故障多样性 (QD Metrics)
    """
    print("--- Plotting Behavior Diversity ---")

    # 检查必要列
    required_cols = ['elapsed_time', 'behavior_count', 'fault_behavior_count', 'phase']
    for col in required_cols:
        if col not in df.columns:
            print(f"[Skip] Column '{col}' not found. Skipping behavior plots.")
            return

    # 排序与筛选
    df_sorted = df.sort_values(by='elapsed_time')
    fuzz_df = df_sorted[df_sorted['phase'] == 'Phase2']
    
    if len(fuzz_df) == 0:
        print("[Info] No Phase2 data found, using all data for behavior plot.")
        fuzz_df = df_sorted

    fuzz_df['time_hours'] = fuzz_df['elapsed_time'] / 3600.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # --- Plot 1: Total Behaviors ---
    ax1.plot(fuzz_df['time_hours'], fuzz_df['behavior_count'], 
             color='#000080', linewidth=2.5, label='Total Behaviors') # Navy Blue
    ax1.fill_between(fuzz_df['time_hours'], fuzz_df['behavior_count'], 
                     color='#000080', alpha=0.1)
    ax1.set_ylabel('# Behaviours', fontsize=16)
    ax1.set_title('Behavior Space Coverage', fontsize=16)
    ax1.grid(True)
    ax1.legend(loc='upper left', frameon=False)

    # --- Plot 2: Faulty Behaviors ---
    ax2.plot(fuzz_df['time_hours'], fuzz_df['fault_behavior_count'], 
             color='#8B0000', linewidth=2.5, label='Faulty Behaviors') # Dark Red
    ax2.fill_between(fuzz_df['time_hours'], fuzz_df['fault_behavior_count'], 
                     color='#8B0000', alpha=0.1)
    ax2.set_xlabel('Time (Hours)', fontsize=16)
    ax2.set_ylabel('# Faulty Behaviours', fontsize=16)
    ax2.set_title('Fault Diversity (QD)', fontsize=16)
    ax2.grid(True)
    ax2.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "qd_diversity_metrics.png")
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Saved: {save_path}")
    plt.close()


def plot_spatial_distribution(df, output_dir=".", town_name="Town01"):
    """
    来自 plot1.py: 绘制失效案例的空间分布散点图
    """
    print("--- Plotting Spatial Distribution ---")

    if 'final_x' not in df.columns or 'final_y' not in df.columns:
        print("[Skip] Missing coordinate columns. Skipping spatial plot.")
        return

    # 筛选未成功的案例 (Failure)
    failures = df[(df['success'] == False)]
    
    # 优先展示 Phase 2
    phase2_failures = failures[failures['phase'] == 'Phase2']
    
    if len(phase2_failures) > 0:
        plot_data = phase2_failures
        label_text = "Distinct Failures (Phase 2)"
    else:
        print("[Info] No failures in Phase2 yet, showing all phases.")
        plot_data = failures
        label_text = "Distinct Failures (All Phases)"

    if len(plot_data) == 0:
        print("[Warning] No failures found in log. Skipping spatial plot.")
        return

    plt.figure(figsize=(8, 8))
    
    plt.scatter(plot_data['final_x'], plot_data['final_y'], 
                c='#008000',      # Green
                marker='o',       
                s=50,             
                alpha=0.7,        
                edgecolors='k',   
                linewidth=0.5,
                label=label_text)

    # 根据城镇名称设置坐标范围
    if town_name == "Town01":
        plt.xlim(-20, 420)
        plt.ylim(-20, 350)
    elif town_name == "Town02":
        plt.xlim(-20, 200)
        plt.ylim(-20, 320)
    else:
        plt.autoscale()

    plt.grid(True)
    plt.xlabel('X Position (m)', fontsize=16)
    plt.ylabel('Y Position (m)', fontsize=16)
    plt.title(f'Spatial Distribution of Failures ({town_name})', fontsize=16)
    plt.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "spatial_diversity_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Saved: {save_path}")
    plt.close()

# ==========================================
# 3. 主程序逻辑
# ==========================================

def get_latest_csv(base_dir="./results", default_csv="summary.csv"):
    """自动查找最新的 summary.csv"""
    target_csv = default_csv
    
    if os.path.exists(base_dir):
        # 获取所有子目录
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if subdirs:
            # 找到修改时间最新的目录
            latest_subdir = max(subdirs, key=os.path.getmtime)
            found_csv = os.path.join(latest_subdir, "summary.csv")
            if os.path.exists(found_csv):
                target_csv = found_csv
                print(f"[Info] Found latest log in results: {target_csv}")
    
    return target_csv

if __name__ == "__main__":
    # 1. 获取 CSV 路径
    csv_path = get_latest_csv()
    
    # 2. 读取数据 (只读一次)
    if not os.path.exists(csv_path):
        print(f"[Error] File not found: {csv_path}")
        exit(1)
        
    try:
        df = pd.read_csv(csv_path)
        print(f"[Info] Data loaded successfully. Rows: {len(df)}")
    except Exception as e:
        print(f"[Error] Could not read CSV: {e}")
        exit(1)

    # 3. 依次调用绘图函数
    # 获取 CSV 所在的目录作为输出目录，如果找不到则输出到当前目录
    output_directory = os.path.dirname(csv_path) if os.path.dirname(csv_path) else "."
    
    plot_crash_curve(df, output_directory)
    plot_behavior_diversity(df, output_directory)
    plot_spatial_distribution(df, output_directory, town_name="Town01")
    
    print("\n[Done] All plots generated.")