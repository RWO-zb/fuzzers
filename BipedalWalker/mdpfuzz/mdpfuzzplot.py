import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.manifold import TSNE
from collections import Counter
import os
import time
import ast 

# --- 1. 配置 ---
LOG_FILE = 'rt_10_0.01_0.01_0_logs.txt' 

# [修改] 数据行数限制
# 如果是 Random Fuzzer 或需要对齐评估预算，设为 330000；
# 如果需要分析所有数据，请设为 None
ROW_LIMIT = None

PLOT_1_FILE = 'rt_crashes_over_unique_inputs.png'
PLOT_2_FILE = 'rt_full_input_space_tsne.png'
PLOT_3_FILE = 'rt_crash_generation_histogram.png'
PLOT_4_FILE = 'rt_crashes_over_time.png'
PLOT_5_FILE = 'rt_behaviour_coverage_heatmap.png' 

# --- 2. 核心辅助函数 ---

def load_and_prepare_data(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        return None
    
    try:
        print(f"正在从 {file_path} 加载原始日志数据...")
        
        # 读取 CSV
        df = pd.read_csv(
            file_path, 
            delimiter=';', 
            on_bad_lines='skip', 
            skipinitialspace=True
        )
        
        # [修改] 如果设置了行数限制，在此处截断数据
        if ROW_LIMIT is not None:
            print(f"   >>> 注意：已启用数据限制，仅保留前 {ROW_LIMIT} 条记录 <<<")
            df = df.iloc[:ROW_LIMIT]

        if df['Oracle'].dtype == 'object':
            df['Oracle'] = df['Oracle'].map({'True': True, 'False': False, 'None': None})
        
        df['is_crash'] = (df['Oracle'] == True)
        
        # 处理数值列
        for col in ['Sensitivity', 'Coverage', 'CoverageTime', 'RunTime', 'BD_Distance', 'BD_MeanAngle']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"原始日志加载完成，当前处理记录数: {len(df)} 条。")
        return df
        
    except Exception as e:
        print(f"加载或处理 CSV 文件时出错: {e}")
        return None

def deduplicate_log(original_data_df):
    if original_data_df is None: return None
    try:
        # 基于 Input 去重
        unique_df = original_data_df.drop_duplicates(subset=['Input'], keep='first')
        unique_df = unique_df.reset_index(drop=True)
        return unique_df
    except KeyError:
        return None
    except Exception as e:
        return None

# --- 3. 绘图函数 ---

def plot_crashes_over_time(unique_log_df):
    if unique_log_df is None: return
    print("\n[图表 1] 正在生成 '独特崩溃 vs 独特输入' 图...")
    cumulative_crashes = unique_log_df['is_crash'].cumsum()
    
    plt.figure(figsize=(12, 7))
    plt.plot(cumulative_crashes)
    plt.title('Total Unique Crashes Found Over Time')
    plt.xlabel('Number of Unique Inputs Explored')
    plt.ylabel('Cumulative Number of Unique Crashes')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(PLOT_1_FILE)
    plt.close()

def plot_crash_generation_histogram(unique_log_df):
    if unique_log_df is None: return
    print("\n[图表 3] 正在生成 '崩溃代数直方图'...")
    crash_data = unique_log_df[unique_log_df['is_crash'] == True]
    if crash_data.empty: return

    try:
        crash_generations = crash_data['Generation'].astype(int)
        generation_counts = Counter(crash_generations)
        generations = sorted(generation_counts.keys())
        counts = [generation_counts[g] for g in generations]
        
        plt.figure(figsize=(12, 7))
        plt.bar(generations, counts, color='red', alpha=0.7)
        plt.title('Histogram of Unique Crash Generations')
        plt.xlabel('Mutation Generation')
        plt.ylabel('Number of Unique Crashing Inputs')
        plt.savefig(PLOT_3_FILE)
        plt.close()
    except Exception as e:
        print(f"Plotting error: {e}")

def plot_crashes_over_wallclock_time(unique_log_df):
    if unique_log_df is None or 'RunTime' not in unique_log_df.columns: return
    print("\n[图表 4] 正在生成 '独特崩溃 vs 时间' 图...")
    crash_df = unique_log_df[unique_log_df['is_crash'] == True].copy()
    if crash_df.empty: return

    start_time = unique_log_df['RunTime'].min()
    crash_times_hours = (crash_df['RunTime'] - start_time) / 3600.0
    crash_times_hours = crash_times_hours.sort_values()
    cumulative_counts = np.arange(1, len(crash_times_hours) + 1)
    
    plt.figure(figsize=(12, 7))
    plt.step(crash_times_hours, cumulative_counts, where='post', color='darkorange', linewidth=2)
    plt.fill_between(crash_times_hours, cumulative_counts, step='post', color='darkorange', alpha=0.1)
    plt.title('Cumulative Unique Crashes vs. Time')
    plt.xlabel('Time Elapsed (hours)')
    plt.ylabel('Cumulative Number of Unique Crashes')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.savefig(PLOT_4_FILE)
    plt.close()

def calculate_behaviour_diversity(unique_log_df, grid_size=(50, 50)):
    """ 计算基于 2D 网格 (Distance, Hull Angle) 的行为多样性 (覆盖率)。 """
    print(f"\n{'='*40}\n Behaviour Diversity Analysis (QD-Fuzz)\n{'='*40}")
    
    if 'BD_Distance' not in unique_log_df.columns or 'BD_MeanAngle' not in unique_log_df.columns:
        print("Warning: 'BD_Distance' or 'BD_MeanAngle' columns not found. Skipping BD analysis.")
        return

    valid_df = unique_log_df.dropna(subset=['BD_Distance', 'BD_MeanAngle'])
    dists = valid_df['BD_Distance'].values
    angles = valid_df['BD_MeanAngle'].values
    
    if len(dists) == 0:
        print("No valid BD data found.")
        return

    min_dist, max_dist = dists.min(), dists.max()
    min_angle, max_angle = angles.min(), angles.max()
    
    dist_bins = np.linspace(min_dist, max_dist, grid_size[0] + 1)
    angle_bins = np.linspace(min_angle, max_angle, grid_size[1] + 1)
    
    dist_indices = np.clip(np.digitize(dists, dist_bins) - 1, 0, grid_size[0] - 1)
    angle_indices = np.clip(np.digitize(angles, angle_bins) - 1, 0, grid_size[1] - 1)
    
    filled_bins = set(zip(dist_indices, angle_indices))
    
    print(f"Total Unique Inputs Analyzed: {len(valid_df)}")
    print(f"Occupied Bins: {len(filled_bins)} / {grid_size[0] * grid_size[1]}")
    print(f"Coverage Ratio: {len(filled_bins) / (grid_size[0] * grid_size[1]):.4%}")
    
    # 绘图
    heatmap = np.zeros(grid_size)
    for i in range(len(dists)):
        heatmap[dist_indices[i], angle_indices[i]] += 1
        
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log1p(heatmap).T, origin='lower', aspect='auto', cmap='viridis', 
               extent=[min_dist, max_dist, min_angle, max_angle])
    plt.colorbar(label='Log(Count)')
    plt.title(f'Behaviour Space Coverage (Filled Bins: {len(filled_bins)})')
    plt.xlabel('Distance')
    plt.ylabel('Mean Hull Angle')
    plt.savefig(PLOT_5_FILE)
    print(f"Saved {PLOT_5_FILE}")
    plt.close()

# --- 4. 主函数 ---

def main():
    print("--- Fuzzer 日志分析脚本 ---")
    start_time = time.time()
    
    original_log_data_df = load_and_prepare_data(LOG_FILE)
    if original_log_data_df is None: return

    # [新增] 统计并打印不去重的 crash 数
    total_raw_crashes = original_log_data_df['is_crash'].sum()
    print(f"\n{'='*35}")
    print(f"总崩溃数量 (Total Raw Crashes, 不去重): {total_raw_crashes}")
    print(f"{'='*35}")

    unique_log_df = deduplicate_log(original_log_data_df)
    if unique_log_df is None: return

    # 统计并打印独特崩溃数量
    unique_crash_count = unique_log_df['is_crash'].sum()
    print(f"\n{'='*35}")
    print(f"独特崩溃数量 (Unique Crashes): {unique_crash_count}")
    print(f"{'='*35}")

    # 执行绘图
    #plot_crashes_over_time(unique_log_df)
    #plot_crash_generation_histogram(unique_log_df)
    #plot_crashes_over_wallclock_time(unique_log_df)
    #calculate_behaviour_diversity(unique_log_df)

    end_time = time.time()
    print(f"\n分析完成！总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()