import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置区域 ---
# 请修改为您实际生成的日志文件路径
LOG_FILE = 'fuzzer_10_0.01_0.01_0_logs.txt' 

# 输出图片文件名
PLOT_BD_CURVE = 'diversity_curve_bd.png'
PLOT_FD_CURVE = 'diversity_curve_fd.png'
PLOT_SC_CURVE = 'diversity_curve_sc.png'
PLOT_ROOT_SEED_HIST = 'root_seed_crash_counts.png'

GRID_SIZE = (50, 50)

def load_data(file_path):
    print(f"Loading log data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return None
        
    filename = os.path.basename(file_path).lower()
    
    try:
        # 使用 python 引擎并跳过错误行，以兼容可能的不完整写入
        df = pd.read_csv(
            file_path, 
            delimiter=';', 
            engine='python', 
            on_bad_lines='skip', 
            skipinitialspace=True
        )
        df.columns = df.columns.str.strip()

        # 处理 Oracle 列
        if 'Oracle' in df.columns and df['Oracle'].dtype == 'object':
            df['Oracle'] = df['Oracle'].astype(str).map({'True': True, 'False': False, 'None': False}).fillna(False)
            
        # 寻找 Generation 列
        gen_col = None
        for col in df.columns:
            if col.lower() == 'generation':
                gen_col = col
                break
        
        # 处理 RootID 列
        if 'RootID' in df.columns:
             df['RootID'] = pd.to_numeric(df['RootID'], errors='coerce')

        # [核心修改] 过滤数据，只保留 Fuzz 阶段
        if gen_col:
            df[gen_col] = pd.to_numeric(df[gen_col], errors='coerce')
            
            original_count = len(df)
            # 只保留 Generation > 0 的行
            df = df[ (df[gen_col].notna()) & (df[gen_col] > 0) ]
            filtered_count = len(df)
            
            print(f"Filtered out {original_count - filtered_count} entries (Generation 0 / Initialization).")
            print(f"Remaining entries for Fuzz stage: {filtered_count}")
            
            if filtered_count == 0:
                print("Warning: No data left after filtering Generation > 0.")
                print("If you are analyzing Random Testing (RT) logs, Generation is always 0.")

        # 处理数值列
        numeric_cols = ['BD_Distance', 'BD_MeanAngle']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(subset=numeric_cols, inplace=True)
        print(f"Loaded {len(df)} valid entries for plotting.")
        return df
    except Exception as e:
        print(f"Error loading log data: {e}")
        return None
        
   

def get_bin_index(value, min_val, max_val, grid_size_dim):
    if max_val <= min_val:
        return 0
    idx = int((value - min_val) / (max_val - min_val) * grid_size_dim)
    return min(max(idx, 0), grid_size_dim - 1)

def calculate_cumulative_trends(df):
    all_dists = df['BD_Distance'].values
    all_angles = df['BD_MeanAngle'].values

    if len(all_dists) == 0:
        print("Warning: No valid behaviour data found.")
        return None

    min_dist, max_dist = np.min(all_dists), np.max(all_dists) + 1e-5
    min_angle, max_angle = np.min(all_angles), np.max(all_angles) + 1e-5
    
    print(f"Global Range - Dist: [{min_dist:.2f}, {max_dist:.2f}], Angle: [{min_angle:.2f}, {max_angle:.2f}]")

    bd_filled_bins = set()
    fd_crash_bins = set()
    unique_states = set()
    
    bd_trend = []
    fd_trend = []
    sc_trend = []
    
    for row in df.itertuples(index=False):
        inp_str = getattr(row, 'Input', None)
        if inp_str:
            unique_states.add(inp_str)
        sc_trend.append(len(unique_states))

        d = getattr(row, 'BD_Distance', None)
        a = getattr(row, 'BD_MeanAngle', None)
        is_crash = getattr(row, 'Oracle', False)
        
        d_idx = get_bin_index(d, min_dist, max_dist, GRID_SIZE[0])
        a_idx = get_bin_index(a, min_angle, max_angle, GRID_SIZE[1])
        bin_loc = (d_idx, a_idx)
        
        bd_filled_bins.add(bin_loc)
        
        if is_crash:
            fd_crash_bins.add(bin_loc)
        
        bd_trend.append(len(bd_filled_bins))
        fd_trend.append(len(fd_crash_bins))
        
    return {
        'x_axis': range(1, len(df) + 1),
        'bd_trend': bd_trend,
        'fd_trend': fd_trend,
        'sc_trend': sc_trend
    }

def plot_curve(x, y, title, ylabel, filename, color, label_prefix):
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, y, linewidth=2, color=color, label=label_prefix)
    plt.fill_between(x, y, color=color, alpha=0.1)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Number of Fuzzing Iterations (Gen > 0)', fontsize=12) # 修改了 X 轴标签
    plt.ylabel(ylabel, fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0, right=len(x))
    plt.ylim(bottom=0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")
    plt.close()

def plot_root_seed_distribution_from_log(df, filename):
    """
    直接从 Log 的 RootID 列绘制初始种子崩溃分布图。
    注意：此时 df 已经过滤掉了 Generation 0。
    这意味着这里统计的是：在 Fuzz 阶段产生的崩溃，分别归属于哪个初始种子。
    """
    print("-" * 30)
    print("Starting Root Seed Analysis from Log (Fuzz Stage Only)...")
    
    if 'RootID' not in df.columns:
        print("Error: 'RootID' column not found in logs. Please run the modified fuzzer first.")
        return

    # 1. 筛选出发生 Crash 的行
    # df 已经是 fuzz stage 了，所以这里统计的是变异出的子代导致的 crash
    crash_df = df[ (df['Oracle'] == True) & (df['RootID'].notna()) ]
    
    if crash_df.empty:
        print("No crashes found in the fuzz stage data.")
        return

    # 2. 统计每个 RootID 的出现次数
    root_counts = crash_df['RootID'].value_counts().sort_values(ascending=False)
    
    if root_counts.empty:
        print("No valid RootIDs found for crashes.")
        return

    print(f"Found {len(root_counts)} unique root seeds contributing to {len(crash_df)} crashes in Fuzz stage.")

    # 3. 绘图
    labels = [f"Seed {int(rid)}" for rid in root_counts.index]
    counts = root_counts.values
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, counts, color='#27ae60', alpha=0.8, edgecolor='black')
    
    plt.title(f'Crash Distribution by Initial Seed (Fuzz Stage Only)\nTotal Fuzz Crashes: {sum(counts)}', fontsize=14)
    plt.xlabel('Initial Seed ID', fontsize=12)
    plt.ylabel('Number of Derived Crashes', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 在柱子上方标注具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved root seed distribution plot to {filename}")
    plt.close()


def main():
    df = load_data(LOG_FILE)
    if df is None or df.empty:
        print("Dataframe is empty. Aborting plotting.")
        return

    is_random = 'random' in os.path.basename(LOG_FILE).lower()
    label_name = 'Random' if is_random else 'MDPFuzz'
    print(f"Generating plots for: {label_name} (Fuzz Stage Only)")

    # 1. 原有的多样性曲线
    trends = calculate_cumulative_trends(df)
    if trends:
        x = trends['x_axis']

        plot_curve(
            x, trends['bd_trend'], 
            title=f'Behaviour Diversity Growth ({label_name} - Fuzz Stage)', 
            ylabel='Cumulative Covered Bins', 
            filename=PLOT_BD_CURVE,
            color='#8e44ad',
            label_prefix=label_name
        )
        
        plot_curve(
            x, trends['fd_trend'], 
            title=f'Fault Diversity Growth ({label_name} - Fuzz Stage)', 
            ylabel='Cumulative Covered Crash Bins', 
            filename=PLOT_FD_CURVE,
            color='#c0392b',
            label_prefix=label_name
        )
        
        plot_curve(
            x, trends['sc_trend'], 
            title=f'State Coverage Growth ({label_name} - Fuzz Stage)', 
            ylabel='Cumulative Unique Inputs', 
            filename=PLOT_SC_CURVE,
            color='#2980b9',
            label_prefix=label_name
        )
    
    # 2. 初始种子柱状图 (从日志直接读取)
    plot_root_seed_distribution_from_log(df, PLOT_ROOT_SEED_HIST)

    print("\nAll analysis completed.")

if __name__ == "__main__":
    main()