import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# 请将此处修改为您实际的 CSV 文件路径
DATA_FILE = '1774592517.034981_data.csv'

# [修改] 所有输出文件名增加 _fuzz_only 后缀
PLOT_BD_CURVE = 'qdfuzz_behaviour_diversity_curve_fuzz_only.png'
PLOT_FD_CURVE = 'qdfuzz_fault_diversity_curve_fuzz_only.png'
PLOT_SC_CURVE = 'qdfuzz_state_coverage_curve_fuzz_only.png'
PLOT_SEED_STAT = 'qdfuzz_seed_crash_stats_fuzz_only.png' 

GRID_SIZE = (50, 50)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    df = pd.read_csv(file_path)   
    # 必须包含 mutation_count 以便区分阶段
    required_cols = ['behavior0', 'behavior1', 'is_faulty', 'input', 'elapsed_time', 'seed_id', 'mutation_count']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in CSV.")
            return None
        
        if 'elapsed_time' in df.columns:
            print("Sorting data by elapsed_time to restore execution order...")
            df = df.sort_values(by='elapsed_time').reset_index(drop=True)
        else:
            print("Warning: 'elapsed_time' not found. Data order might not reflect execution order.")

        print(f"Loaded {len(df)} entries.")
        return df

def get_bin_index(value, min_val, max_val, grid_size_dim):
    if max_val <= min_val:
        return 0
    idx = int((value - min_val) / (max_val - min_val) * grid_size_dim)
    return min(max(idx, 0), grid_size_dim - 1)

def calculate_cumulative_trends(df):
    # 注意：这里的 df 已经是经过筛选的 Fuzz Phase 数据
    all_b0 = df['behavior0'].values
    all_b1 = df['behavior1'].values
    
    if len(all_b0) == 0:
        print("Warning: No behavior data found in the filtered dataset.")
        return None
    
    # 计算 Range 时最好基于该阶段的数据，或者如果您想基于全局Range，可以手动指定
    # 这里基于当前数据计算 Range
    min_b0, max_b0 = np.min(all_b0), np.max(all_b0) + 1e-5
    min_b1, max_b1 = np.min(all_b1), np.max(all_b1) + 1e-5
    print(f"Fuzz Phase Range - Behavior0: [{min_b0:.2f}, {max_b0:.2f}], Behavior1: [{min_b1:.2f}, {max_b1:.2f}]")

    bd_filled_bins = set()     
    fd_crash_bins = set()      
    unique_states = set()      
    
    bd_trend = []
    fd_trend = []
    sc_trend = []
    
    # 遍历数据，计算累积值
    # 注意：这里的累积是从 Fuzz 阶段开始从 0 累积的
    for row in df.itertuples(index=False):
        inp_str = getattr(row, 'input', None)
        if inp_str:
            unique_states.add(inp_str)
        sc_trend.append(len(unique_states))

        b0 = getattr(row, 'behavior0', None)
        b1 = getattr(row, 'behavior1', None)
        is_faulty = getattr(row, 'is_faulty', False)
        
        b0_idx = get_bin_index(b0, min_b0, max_b0, GRID_SIZE[0])
        b1_idx = get_bin_index(b1, min_b1, max_b1, GRID_SIZE[1])
        bin_loc = (b0_idx, b1_idx)
        
        bd_filled_bins.add(bin_loc)
        
        if is_faulty:
            fd_crash_bins.add(bin_loc)
        
        bd_trend.append(len(bd_filled_bins))
        fd_trend.append(len(fd_crash_bins))
        
    return {
        'x_axis': range(1, len(df) + 1), 
        'bd_trend': bd_trend,
        'fd_trend': fd_trend,
        'sc_trend': sc_trend
    }

def plot_curve(x, y, title, ylabel, filename, color):
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, y, linewidth=2, color=color, label='QDFuzz (Fuzz Phase)')
    plt.fill_between(x, y, color=color, alpha=0.1)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Number of Fuzzing Executions', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0, right=len(x))
    plt.ylim(bottom=0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")
    plt.close()

def plot_seed_crash_statistics(df, filename):
    print("\nCalculating Seed Crash Statistics (Fuzz Phase)...")
    
    # 筛选导致 Crash 的数据 (df 已经是 Fuzz Phase Only 了)
    crashes = df[(df['is_faulty'] == True) & (df['seed_id'] != -1)]
    
    if crashes.empty:
        print("No crashes found in the filtered Fuzzing Phase data.")
        return

    # 统计每个 seed_id 出现的次数
    seed_counts = crashes['seed_id'].value_counts().sort_index()
    
    unique_seeds_count = len(seed_counts)
    total_crashes = len(crashes)
    
    print(f"Total Seeds Producing Crashes in Fuzz Phase: {unique_seeds_count}")
    print(f"Total Crashes Found in Fuzz Phase: {total_crashes}")
    
    # 绘图
    plt.figure(figsize=(12, 6))
    
    seed_counts.plot(kind='bar', color='#e74c3c', width=0.8)
    
    plt.title(f'Crashes per Initial Seed (Fuzz Phase Only)\n(Faulty Seeds: {unique_seeds_count}, Total Crashes: {total_crashes})', fontsize=14, fontweight='bold')
    plt.xlabel('Initial Seed ID', fontsize=12)
    plt.ylabel('Number of Crashes (Offspring)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 调整X轴标签显示
    if unique_seeds_count > 50:
        plt.xticks(ticks=range(0, unique_seeds_count, 5), rotation=90)
    else:
        plt.xticks(rotation=45 if unique_seeds_count > 20 else 0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved fuzz-phase seed crash statistics to {filename}")

def main():
    df = load_data(DATA_FILE)
    if df is None:
        return

    # [关键修改] 全局过滤：只保留 mutation_count > 0 的数据（即 Fuzzing 阶段）
    print("\nFiltering data for Fuzzing Phase Only (mutation_count > 0)...")
    df_fuzz = df[df['mutation_count'] > 0].copy()
    
    if df_fuzz.empty:
        print("Error: No data found for Fuzzing Phase (check if mutation_count is logged correctly).")
        return
        
    print(f"Data reduced from {len(df)} (Total) to {len(df_fuzz)} (Fuzz Only).")

    # 1. 生成三条曲线图 (使用 Fuzz Only 数据)
    trends = calculate_cumulative_trends(df_fuzz)
    if trends:
        x = trends['x_axis']

        plot_curve(
            x, trends['bd_trend'], 
            title='Behaviour Diversity Growth (Fuzz Phase Only)', 
            ylabel='Cumulative Covered Bins', 
            filename=PLOT_BD_CURVE,
            color='#f39c12' 
        )
        
        plot_curve(
            x, trends['fd_trend'], 
            title='Fault Diversity Growth (Fuzz Phase Only)', 
            ylabel='Cumulative Covered Crash Bins', 
            filename=PLOT_FD_CURVE,
            color='#d35400'
        )
        
        plot_curve(
            x, trends['sc_trend'], 
            title='State Coverage Growth (Fuzz Phase Only)', 
            ylabel='Cumulative Unique Inputs', 
            filename=PLOT_SC_CURVE,
            color='#27ae60' 
        )

    # 2. 生成种子统计柱状图 (使用 Fuzz Only 数据)
    plot_seed_crash_statistics(df_fuzz, PLOT_SEED_STAT)

    print("\nAll fuzz-phase curves generated successfully.")

if __name__ == "__main__":
    main()