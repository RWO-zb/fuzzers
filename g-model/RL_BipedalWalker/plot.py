import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import time
# import argparse  <-- 已移除

# --- 1. 配置 ---

# !!! 请在这里修改为您要分析的文件的“相对路径” !!!
# 假设您从 RL_BipedalWalker 文件夹运行此脚本，
# 那么您的路径应该以 "results/" 开头。
LOG_FILE_PATH = "all_test_cases_log.pkl"  # <-- 在这里修改文件名

# 这些将是保存的图表文件名generative+novelty_50_all_test_cases_log.pkl
PLOT_1_FILE = 'test_gen_crash_trend.png'
PLOT_2_FILE = 'test_gen_input_space_tsne.png'

# --- 2. 核心辅助函数 (已修改以适应 test_gen.py 的日志) ---

def load_data(file_path):
    """加载 all_test_cases_log.pkl 文件"""
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        print(f"请确保 'LOG_FILE_PATH' (在 {__file__} 的第 11 行) 设置了正确的文件路径。")
        return None
    
    try:
        print(f"正在从 {file_path} 加载原始日志数据...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"原始日志加载完成，总共 {len(data)} 条记录。")
        return data
    except Exception as e:
        print(f"加载 pickle 文件时出错: {e}")
        return None

def deduplicate_log(original_log_data):
    """
    (新函数)
    根据 'input' 字段对日志进行去重，只保留第一次出现的条目。
    这对于绘制“独特输入”图表至关重要。
    """
    print("正在根据 'input' 对日志进行去重（保留首次出现）...")
    
    seen_inputs = set()
    deduplicated_log = []
    
    for entry in original_log_data:
        # 'input' 是一个列表，列表不能被哈希。将其转换为元组(tuple)。
        try:
            input_tuple = tuple(entry["input"])
        except (KeyError, TypeError):
            print("警告: 发现格式不正确的日志条目，已跳过。")
            continue
            
        # 核心去重逻辑
        if input_tuple not in seen_inputs:
            seen_inputs.add(input_tuple)
            deduplicated_log.append(entry)

    print(f"去重完成。总共找到 {len(deduplicated_log)} 个独特的 'input'。")
    
    if not deduplicated_log:
        print("错误：未能从日志中检测到任何有效的 'input'。")
        return None

    return deduplicated_log


# --- 3. 图表1：崩溃趋势图 (已修改) ---

def plot_crash_trend(deduplicated_log):
    """
    (已修改)
    绘制随发现的“独特输入”数量而变化的“独特崩溃”累积数量。
    使用 'is_crash' 键。
    """
    print(f"\n[图表 1] 正在计算崩溃趋势...")
    
    cumulative_crashes_list = []
    current_crash_count = 0
    
    # deduplicated_log 已经按首次出现排序
    for i, entry in enumerate(deduplicated_log):
        # 使用 'is_crash' 键 (来自 test_gen.py)
        if entry.get('is_crash', False):
            current_crash_count += 1
        
        cumulative_crashes_list.append(current_crash_count)
            
    if not cumulative_crashes_list:
        print("[图表 1] 未找到可绘制的崩溃趋势数据。")
        return

    print(f"正在绘制 {len(cumulative_crashes_list)} 个独特输入...")
    
    iterations = range(1, len(cumulative_crashes_list) + 1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, cumulative_crashes_list, label='Cumulative Unique Crashes', color='red', linewidth=2)
    plt.fill_between(iterations, cumulative_crashes_list, color='red', alpha=0.1)
    
    plt.title('Unique Crashes Found vs. Unique Inputs Discovered')
    plt.xlabel('Number of Unique Inputs Discovered (by First Appearance)')
    plt.ylabel('Cumulative Number of Unique Crashing Inputs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    
    try:
        plt.savefig(PLOT_1_FILE)
        print(f"[图表 1] 已保存到: {PLOT_1_FILE}")
    except Exception as e:
        print(f"[图表 1] 保存图表时出错: {e}")
    plt.close() # 释放内存


# --- 4. 图表2：t-SNE 空间图 (已修改) ---

def run_tsne(data, n_samples):
    """
    (保持不变)
    对 (N, 15) 的数据运行 t-SNE 降维到 2 维
    """
    if n_samples < 50:
        print(f"数据点太少 ({n_samples}个)，t-SNE 可能效果不佳。")
        perplexity_value = max(5, n_samples - 1)
    else:
        perplexity_value = min(30, n_samples - 1)
        
    print(f"正在对 {n_samples} 个独特的输入（{data.shape[1]}维）运行 t-SNE (Perplexity={perplexity_value})...")
    
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

def plot_full_space(deduplicated_log):
    """
    (已修改)
    绘制 t-SNE 结果的 2D 散点图，用两种颜色区分
    使用 'input' 和 'is_crash' 键。
    """
    print(f"\n[图表 2] 正在准备 t-SNE 数据...")
    
    all_data_list = []
    labels_list = []
    
    for entry in deduplicated_log:
        try:
            # 'input' 是一个列表，直接使用
            all_data_list.append(entry["input"])
            # 'is_crash' 是一个布尔值
            labels_list.append(1 if entry["is_crash"] else 0) # 1 = Crash, 0 = No Crash
        except KeyError:
            continue
        
    if not all_data_list:
        print("[图表 2] 未找到可用于 t-SNE 的数据。")
        return

    # 将列表的列表转换为 (N, 15) 的 Numpy 数组
    all_data = np.array(all_data_list)
    labels = np.array(labels_list)
    
    # 运行 t-SNE
    tsne_results = run_tsne(all_data, all_data.shape[0])
    
    # 绘制
    print(f"正在绘制 {tsne_results.shape[0]} 个数据点...")
    crashing_points = tsne_results[labels == 1]
    non_crashing_points = tsne_results[labels == 0]
    
    print(f"  非崩溃点: {non_crashing_points.shape[0]}")
    print(f"  崩溃点: {crashing_points.shape[0]}")
    
    plt.figure(figsize=(12, 10))
    plt.scatter(
        non_crashing_points[:, 0], non_crashing_points[:, 1], 
        c='blue', alpha=0.4, s=10, 
        label=f'Non-Crashing Inputs ({non_crashing_points.shape[0]})'
    )
    if crashing_points.shape[0] > 0:
        plt.scatter(
            crashing_points[:, 0], crashing_points[:, 1], 
            c='red', alpha=0.8, s=15, 
            label=f'Crashing Inputs ({crashing_points.shape[0]})'
        )
    
    plt.title('t-SNE Visualization of Unique Explored Inputs (by First Appearance)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    try:
        plt.savefig(PLOT_2_FILE)
        print(f"[图表 2] 已保存到: {PLOT_2_FILE}")
    except Exception as e:
        print(f"[图表 2] 保存图表时出错: {e}")
    plt.close() # 释放内存


# --- 5. 主函数 (已修改) ---

def main():
    # 命令行解析代码已被移除
    # parser = argparse.ArgumentParser(description="为 test_gen.py 日志绘制分析图表")
    # ...
    # args = parser.parse_args()

    # 1. 加载原始数据
    # 直接使用在脚本顶部定义的 LOG_FILE_PATH 变量
    original_log_data = load_data(LOG_FILE_PATH)
    if not original_log_data:
        return

    # 2. 对日志进行去重
    # (这是新日志格式所必需的)
    deduplicated_log = deduplicate_log(original_log_data)
    if not deduplicated_log:
        print("未能从日志中提取任何有效数据。退出。")
        return
        
    # 3. 运行图表 1
    plot_crash_trend(deduplicated_log)
    
    # 4. 运行图表 2
    plot_full_space(deduplicated_log)
        
    print("\n所有分析和绘图已完成。")

if __name__ == "__main__":
    main()