import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import os
import time
import ast # 用于安全地将字符串 '[3, 2, 1]' 转换为列表 [3, 2, 1]

# --- 1. 配置 ---
# !!! 关键: 请将 'your_data_file.csv' 替换为您实际的文件名 !!!
# 假设您的 Excel 文件已另存为 CSV 格式
LOG_FILE = '1762783158.2151027_data.csv' 

PLOT_1_FILE = 'faults_over_unique_inputs.png'
PLOT_2_FILE = 'full_input_space_tsne.png'
PLOT_3_FILE = 'fault_generation_histogram.png'

# --- 2. 核心辅助函数 (已修改为 Pandas 和 CSV) ---

def parse_input_string(input_str):
    """
    安全地将 "[3, 2, 1]" 格式的字符串转换为 Numpy 数组。
    """
    try:
        # ast.literal_eval 是安全的，它只能解析 Python 字面量 (如列表, 数字)
        vector = np.array(ast.literal_eval(input_str), dtype=np.int32)
        return vector
    except (ValueError, SyntaxError, TypeError):
        # 如果 input_str 为空、NaN 或是无效格式，则返回 None
        return None

def load_and_deduplicate(file_path):
    """
    加载 CSV 文件，并根据 'input' 列进行去重 (保留首次出现)。
    """
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        print(f"请确保此脚本与 '{file_path}' 位于同一文件夹中。")
        return None
    
    try:
        print(f"正在从 {file_path} 加载 CSV 数据...")
        # 使用 pandas 加载 CSV
        df = pd.read_csv(file_path)
        print(f"原始 CSV 加载完成，总共 {len(df)} 条记录。")

        # 确保列名正确 (根据您的图片)
        required_cols = ['input', 'is_faulty', 'mutation_count']
        if not all(col in df.columns for col in required_cols):
            print(f"错误: CSV 文件缺少必要的列。")
            print(f"需要: {required_cols}, 实际有: {df.columns.tolist()}")
            return None

    except Exception as e:
        print(f"加载 CSV 文件时出错: {e}")
        return None

    # 核心去重逻辑: 按 'input' 列去重，保留 'first' (第一次出现)
    print("正在根据 'input' 列对日志进行去重（保留首次出现）...")
    # dropna() 确保我们不会因为空 'input' 值而出错
    deduplicated_df = df.dropna(subset=['input']).drop_duplicates(
        subset=['input'], 
        keep='first'
    ).copy() # .copy() 避免 SettingWithCopyWarning

    print(f"去重完成。总共找到 {len(deduplicated_df)} 个独特的 'input'。")

    # --- 为 t-SNE 准备数据 ---
    # 将 'input' 字符串列转换为 'input_vector' Numpy 数组列
    print("正在将 'input' 字符串转换为 t-SNE 向量...")
    deduplicated_df['input_vector'] = deduplicated_df['input'].apply(parse_input_string)
    
    # 丢弃那些无法解析的行
    original_count = len(deduplicated_df)
    deduplicated_df = deduplicated_df.dropna(subset=['input_vector'])
    if len(deduplicated_df) < original_count:
        print(f"警告: 移除了 {original_count - len(deduplicated_df)} 个 'input' 格式无效或为空的行。")

    # 检查 'is_faulty' 列是否为布尔值 (CSV 中可能是 'TRUE'/'FALSE' 字符串)
    if deduplicated_df['is_faulty'].dtype == 'object':
        # 将 'True', 'true', 'TRUE' (不区分大小写) 转为布尔值 True
        deduplicated_df['is_faulty'] = deduplicated_df['is_faulty'].astype(str).str.lower() == 'true'
    else:
        # 否则假定它是 0/1 或布尔值
        deduplicated_df['is_faulty'] = deduplicated_df['is_faulty'].astype(bool)

    print("数据准备完毕。")
    return deduplicated_df


# --- 3. 图表1：故障趋势图 (已修改) ---

def plot_fault_trend(deduplicated_df):
    """
    绘制随发现的“独特输入”数量而变化的“独特故障”累积数量。
    (仿照 plot_crash_trend)
    """
    print(f"\n[图表 1] 正在计算故障趋势...")
    
    if deduplicated_df.empty:
        print("[图表 1] 未找到可绘制的故障趋势数据。")
        return

    # 'is_faulty' 是布尔值，cumsum() 会自动将其视为 1 (True) 和 0 (False)
    cumulative_faults_list = deduplicated_df['is_faulty'].cumsum()
    
    print(f"正在绘制 {len(cumulative_faults_list)} 个独特输入...")
    
    iterations = range(1, len(cumulative_faults_list) + 1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, cumulative_faults_list, label='Cumulative Unique Faults', color='red', linewidth=2)
    plt.fill_between(iterations, cumulative_faults_list, color='red', alpha=0.1)
    
    plt.title('Unique Faults Found vs. Unique Inputs Discovered')
    plt.xlabel('Number of Unique Inputs Discovered (by First Appearance)')
    plt.ylabel('Cumulative Number of Unique Faulty Inputs')
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

def run_tsne(data, n_samples, n_dimensions):
    """
    (基本不变)
    对 (N, D) 的数据运行 t-SNE 降维到 2 维
    """
    if n_samples < 50:
        print(f"数据点太少 ({n_samples}个)，t-SNE 可能效果不佳。")
        perplexity_value = max(5, n_samples - 1)
    else:
        perplexity_value = min(30, n_samples - 1)
        
    print(f"正在对 {n_samples} 个独特的输入（{n_dimensions}维）运行 t-SNE (Perplexity={perplexity_value})...")
    
    start_time = time.time()
    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=perplexity_value,
        max_iter=1000, # 增加迭代次数以获得更好的收敛
        random_state=42
    )
    tsne_results = tsne.fit_transform(data)
    
    end_time = time.time()
    print(f"t-SNE 运行完成，耗时: {end_time - start_time:.2f} 秒")
    return tsne_results

def plot_full_space(deduplicated_df):
    """
    (已修改)
    绘制 t-SNE 结果的 2D 散点图，用两种颜色区分
    (仿照 plot_full_space)
    """
    print(f"\n[图表 2] 正在准备 t-SNE 数据...")
    
    # 从 DataFrame 中提取准备好的数据
    all_data_list = deduplicated_df['input_vector'].tolist()
    labels_list = deduplicated_df['is_faulty'].tolist() # True = Fault, False = No Fault
        
    if not all_data_list:
        print("[图表 2] 未找到可用于 t-SNE 的数据。")
        return

    # vstack 将列表中的所有 (N,) 数组堆叠成 (N, D) 矩阵
    all_data = np.vstack(all_data_list)
    labels = np.array(labels_list) # 已是布尔值
    
    # 运行 t-SNE
    tsne_results = run_tsne(all_data, all_data.shape[0], all_data.shape[1])
    
    # 绘制
    print(f"正在绘制 {tsne_results.shape[0]} 个数据点...")
    faulty_points = tsne_results[labels == True]
    non_faulty_points = tsne_results[labels == False]
    
    print(f"  非故障点: {non_faulty_points.shape[0]}")
    print(f"  故障点: {faulty_points.shape[0]}")
    
    plt.figure(figsize=(12, 10))
    plt.scatter(
        non_faulty_points[:, 0], non_faulty_points[:, 1], 
        c='blue', alpha=0.4, s=10, 
        label=f'Non-Faulty Inputs ({non_faulty_points.shape[0]})'
    )
    if faulty_points.shape[0] > 0:
        plt.scatter(
            faulty_points[:, 0], faulty_points[:, 1], 
            c='red', alpha=0.8, s=15, 
            label=f'Faulty Inputs ({faulty_points.shape[0]})'
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


# --- 5. 图表3：故障代数直方图 (已修改) ---

def plot_generation_histogram(deduplicated_df):
    """
    (已修改)
    绘制 *独特故障输入* 的代数直方图。
    (仿照 plot_generation_histogram，使用 'mutation_count')
    """
    print(f"\n[图表 3] 正在分析故障代数...")
    
    # 1. 筛选出所有 'is_faulty' 为 True 的独特输入
    faulty_df = deduplicated_df[deduplicated_df['is_faulty'] == True]
    
    # 2. 获取这些故障的 'mutation_count'
    # .values 将其转换为 Numpy 数组，如果为空则为空数组
    fault_generations = faulty_df['mutation_count'].values.astype(int) 
            
    print(f"分析完成。总共找到 {len(fault_generations)} 个独特的故障事件。")
    
    if len(fault_generations) == 0:
        print("[图表 3] 未找到故障代数数据，无法绘图。")
        return

    generation_counts = Counter(fault_generations)
    
    # 找出最大代数
    max_gen = 0
    if generation_counts:
        max_gen = max(generation_counts.keys())
        
    generations = range(0, max_gen + 2) # 从0开始，多显示一格
    counts = [generation_counts.get(gen, 0) for gen in generations]
    
    print("\n--- 独特故障代数统计 ---")
    print(f"  平均故障代数: {np.mean(fault_generations):.2f}")
    print(f"  中位故障代数: {np.median(fault_generations)}")
    print(f"  最大故障代数: {np.max(fault_generations)}")

    plt.figure(figsize=(12, 7))
    plt.bar(generations, counts, color='red', alpha=0.7, zorder=3)
    
    plt.title('Histogram of Unique Fault Generations (by First Appearance)')
    plt.xlabel('Mutation Generation (Depth from Initial Seed)') # 假设 mutation_count 从 0 开始
    plt.ylabel('Number of Unique Faulty Inputs')
    
    step = max(1, (max_gen // 20)) # 动态调整x轴刻度
    plt.xticks(np.arange(0, max_gen + 2, step=step))
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    try:
        plt.savefig(PLOT_3_FILE)
        print(f"[图表 3] 已保存到: {PLOT_3_FILE}")
    except Exception as e:
        print(f"[图表 3] 保存图表时出错: {e}")
    plt.close() # 释放内存

# --- 6. 主函数 (已修改) ---

def main():
    # 1. 加载并去重数据
    # 这一步现在合并了加载、去重和 t-SNE 数据准备
    deduplicated_df = load_and_deduplicate(LOG_FILE)
    
    if deduplicated_df is None or deduplicated_df.empty:
        print("未能从日志中提取任何有效数据。退出。")
        return
        
    # 2. 运行图表 1
    plot_fault_trend(deduplicated_df)
    
    # 3. 运行图表 2
    plot_full_space(deduplicated_df)
    
    # 4. 运行图表 3
    plot_generation_histogram(deduplicated_df)
        
    print("\n所有分析和绘图已完成。")

if __name__ == "__main__":
    main()