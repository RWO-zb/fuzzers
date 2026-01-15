import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import os
import sys

# ==========================================
# 1. 配置与常量
# ==========================================
INPUT_CSV = 'summary.csv'

# 输出图片文件名 (增加 fuzz 前缀以示区别)
PLOT_1_FILE = 'fuzz_failures_over_inputs.png'
PLOT_2_FILE = 'fuzz_input_space_tsne.png'
PLOT_3_FILE = 'fuzz_failure_generation_hist.png'
PLOT_4_FILE = 'fuzz_failures_over_time.png'
PLOT_5_FILE = 'fuzz_behaviour_heatmap.png'

# 绘图风格设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# 2. 数据解析与处理
# ==========================================

def parse_input_features(input_str):
    """
    从 input_post 字符串中提取特征向量用于 t-SNE。
    """
    if pd.isna(input_str) or str(input_str) == "None":
        return None
    
    try:
        # 格式示例: "Ego:[x,y,yaw]|NPCs:(x1,y1)..."
        parts = str(input_str).split('|')
        if len(parts) < 2: return None
        
        # 1. Ego 特征
        ego_part = parts[0].split(':')[1].strip('[]')
        ego_vals = [float(x) for x in ego_part.split(',') if x]
        
        # 2. NPC 特征
        npc_part = parts[1].split(':')[1]
        if not npc_part or npc_part == 'None':
            npc_feats = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            # 去除括号并分割
            raw_nums = npc_part.replace('(', '').replace(')', '').split(',')
            coords = [float(x) for x in raw_nums if x]
            
            if not coords:
                npc_feats = [0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                xs = coords[0::2]
                ys = coords[1::2]
                npc_feats = [
                    float(len(xs)),      # Count
                    np.mean(xs),         # Mean X
                    np.mean(ys),         # Mean Y
                    np.std(xs) if len(xs) > 1 else 0.0, # Std X
                    np.std(ys) if len(ys) > 1 else 0.0  # Std Y
                ]
                
        return np.array(ego_vals + npc_feats)
    except:
        return None

def load_and_process_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"[Error] 文件未找到: {csv_path}")
        return None

    print(f"正在读取数据: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] 读取 CSV 失败: {e}")
        return None

    # --- [关键修改] 过滤 Fuzz 阶段数据 ---
    if 'phase' in df.columns:
        print("正在过滤数据: 仅保留 'Phase2' (Fuzzing Phase)...")
        original_len = len(df)
        df = df[df['phase'] == 'Phase2']
        print(f"过滤结果: {original_len} -> {len(df)} 条记录")
        
        if len(df) == 0:
            print("[Warning] 没有找到 Phase2 的数据！请检查 CSV 文件是否包含 Fuzzing 阶段记录。")
            return None
    else:
        print("[Warning] 未找到 'phase' 列，将使用所有数据。")

    # 按时间排序
    if 'elapsed_time' in df.columns:
        df = df.sort_values(by='elapsed_time')

    processed_data = []
    seen_inputs = set()
    
    print("正在处理数据 (去重 & 特征提取)...")
    
    for _, row in df.iterrows():
        # 1. 获取去重键: input_post
        raw_input = row.get('input_post') 
        
        if pd.isna(raw_input) or str(raw_input) == "None":
            continue
            
        unique_key = str(raw_input).strip()
        
        # 2. 去重逻辑
        if unique_key not in seen_inputs:
            seen_inputs.add(unique_key)
            
            entry = row.to_dict()
            
            # 3. 解析特征
            entry['features'] = parse_input_features(unique_key)
            
            # 4. 定义 "Crash" (Failure): success == False
            is_success = str(entry.get('success', 'False')).lower() == 'true'
            entry['is_crash'] = not is_success  
            
            processed_data.append(entry)
            
    print(f"Fuzz 阶段唯一输入数: {len(processed_data)}")
    print(f"Fuzz 阶段故障数 (Success=False): {sum(1 for d in processed_data if d['is_crash'])}")
    
    return processed_data

# ==========================================
# 3. 绘图函数
# ==========================================

def plot_crash_trend(data):
    """[图表 1] Fuzz 阶段累计故障数 vs 唯一输入数"""
    cumulative_failures = []
    current_count = 0
    
    for entry in data:
        if entry['is_crash']:
            current_count += 1
        cumulative_failures.append(current_count)
        
    if not cumulative_failures:
        print("[Skip] 没有数据用于绘制趋势图")
        return

    x = range(1, len(cumulative_failures) + 1)
    
    plt.figure()
    plt.plot(x, cumulative_failures, label='Cumulative Failures', color='#D62728', linewidth=2.5)
    plt.fill_between(x, cumulative_failures, color='#D62728', alpha=0.1)
    
    plt.title('Fuzzing Failures (Success=False) vs. Unique Inputs')
    plt.xlabel('Number of Unique Inputs Discovered (Phase 2)')
    plt.ylabel('Cumulative Number of Failures')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(PLOT_1_FILE, dpi=300)
    print(f"[Saved] {PLOT_1_FILE}")
    plt.close()

def plot_tsne_space(data):
    """[图表 2] Fuzz 阶段输入空间 t-SNE"""
    valid_data = [d for d in data if d['features'] is not None]
    
    if len(valid_data) < 5:
        print("[Skip] 数据点不足以进行 t-SNE 分析 (需要 >= 5)")
        return

    print(f"正在运行 t-SNE (样本数: {len(valid_data)})...")
    features = np.array([d['features'] for d in valid_data])
    labels = np.array([1 if d['is_crash'] else 0 for d in valid_data])
    
    perplexity_val = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    # Blue: Success
    safe_mask = (labels == 0)
    plt.scatter(tsne_results[safe_mask, 0], tsne_results[safe_mask, 1], 
                c='#1F77B4', alpha=0.4, s=20, label=f'Success ({sum(safe_mask)})')
    
    # Red: Failure
    fail_mask = (labels == 1)
    if sum(fail_mask) > 0:
        plt.scatter(tsne_results[fail_mask, 0], tsne_results[fail_mask, 1], 
                    c='#D62728', alpha=0.9, s=40, marker='x', label=f'Failure ({sum(fail_mask)})')
        
    plt.title('t-SNE Visualization of Fuzzing Input Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_2_FILE, dpi=300)
    print(f"[Saved] {PLOT_2_FILE}")
    plt.close()

def plot_generation_hist(data):
    """[图表 3] 故障发生的变异代数直方图"""
    fail_gens = []
    
    for entry in data:
        if entry['is_crash']: 
            gen = entry.get('mutation_generation')
            try:
                if pd.notna(gen):
                    fail_gens.append(int(float(gen)))
            except:
                continue
                
    if not fail_gens:
        print("[Skip] 没有故障代数数据")
        return

    counts = Counter(fail_gens)
    if not counts: return 

    max_gen = max(counts.keys())
    generations = range(0, max_gen + 1)
    values = [counts.get(g, 0) for g in generations]
    
    plt.figure()
    bars = plt.bar(generations, values, color='#FF7F0E', alpha=0.8, edgecolor='black', width=0.8)
    
    plt.title('Histogram of Fuzzing Failure Generations')
    plt.xlabel('Mutation Generation')
    plt.ylabel('Number of Unique Failures')
    
    if max_gen < 20:
        plt.xticks(generations)
    else:
        step = max(1, max_gen // 10)
        plt.xticks(np.arange(0, max_gen + 1, step))
        
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_3_FILE, dpi=300)
    print(f"[Saved] {PLOT_3_FILE}")
    plt.close()

def plot_time_curve(data):
    """[图表 4] Fuzz 阶段随时间变化的累计故障数"""
    fail_times = []
    
    # 获取 Phase 2 的起始时间，以便归一化（可选，这里保持绝对时间差）
    # 如果想看从 Fuzz 开始的时间，可以减去 data[0]['elapsed_time']
    
    for entry in data:
        if entry['is_crash']:
            t = entry.get('elapsed_time')
            if pd.notna(t):
                fail_times.append(float(t))
                
    if not fail_times:
        return
        
    fail_times.sort()
    # 使用绝对时间（小时）
    times_hours = [t / 3600.0 for t in fail_times]
    counts = list(range(1, len(fail_times) + 1))
    
    plt.figure()
    plt.step(times_hours, counts, where='post', color='#2CA02C', linewidth=2.5)
    plt.fill_between(times_hours, counts, step='post', color='#2CA02C', alpha=0.1)
    
    plt.title('Cumulative Fuzzing Failures vs. Total Time')
    plt.xlabel('Time Elapsed (hours)')
    plt.ylabel('Cumulative Number of Failures')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(PLOT_4_FILE, dpi=300)
    print(f"[Saved] {PLOT_4_FILE}")
    plt.close()

def plot_behavior_heatmap(data, bins=20):
    """[图表 5] Fuzz 阶段行为覆盖率热力图"""
    speeds = []
    steers = []
    
    for entry in data:
        s = entry.get('avg_speed')
        st = entry.get('steer_std')
        
        if pd.notna(s) and pd.notna(st):
            speeds.append(float(s))
            steers.append(float(st))
            
    if not speeds:
        print("[Skip] 没有行为数据")
        return
        
    max_speed = max(15, max(speeds))
    max_steer = max(0.5, max(steers))
    
    plt.figure(figsize=(9, 7))
    
    h = plt.hist2d(speeds, steers, bins=bins, 
                   range=[[0, max_speed], [0, max_steer]],
                   cmap='viridis', 
                   norm=plt.matplotlib.colors.LogNorm())
    
    cbar = plt.colorbar(h[3])
    cbar.set_label('Count (Log Scale)')
    
    heatmap_matrix = h[0]
    covered = np.sum(heatmap_matrix > 0)
    total = bins * bins
    
    plt.title(f'Fuzzing Behaviour Coverage\n(Avg Speed vs Steer Std)\nCoverage: {covered}/{total} ({covered/total:.1%})')
    plt.xlabel('Average Speed (m/s)')
    plt.ylabel('Steering Std Dev')
    
    plt.tight_layout()
    plt.savefig(PLOT_5_FILE, dpi=300)
    print(f"[Saved] {PLOT_5_FILE}")
    plt.close()

# ==========================================
# 主程序入口
# ==========================================

if __name__ == "__main__":
    print("--- 开始绘制 Fuzz 阶段 (Phase 2 Only) 图表 ---")
    
    data = load_and_process_data(INPUT_CSV)
    
    if data:
        plot_crash_trend(data)
        plot_tsne_space(data)
        plot_generation_hist(data)
        plot_time_curve(data)
        plot_behavior_heatmap(data)
        
        print("\n所有图表绘制完成。")
    else:
        print("\n无法加载数据，请检查 summary.csv 是否包含 Phase2 数据。")