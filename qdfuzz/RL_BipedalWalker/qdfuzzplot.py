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
# 对应 bw_framework.py 生成的 {timestamp}_data.csv
LOG_FILE = '1765639810.5339673_data.csv' 

PLOT_1_FILE = 'faults_over_unique_inputs.png'
PLOT_2_FILE = 'full_input_space_tsne.png'
PLOT_3_FILE = 'fault_generation_histogram.png'
PLOT_4_FILE = 'faults_over_time.png'  # <--- [新增] 随时间变化的图表文件名

# --- 2. 核心辅助函数 ---

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

        # 确保列名正确
        # [修改]: 添加了 'elapsed_time'
        required_cols = ['input', 'is_faulty', 'mutation_count', 'elapsed_time']
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

    # 确保 elapsed_time 是数值类型
    deduplicated_df['elapsed_time'] = pd.to_numeric(deduplicated_df['elapsed_time'], errors='coerce').fillna(0.0)

    print("数据准备完毕。")
    return deduplicated_df


# --- 3. 图表1：故障趋势图 ---

def plot_fault_trend(deduplicated_df):
    """
    绘制随发现的“独特输入”数量而变化的“独特故障”累积数量。
    """
    print(f"\n[图表 1] 正在计算故障趋势(按输入数量)...")
    
    if deduplicated_df.empty:
        print("[图表 1] 未找到可绘制的数据。")
        return

    # 'is_faulty' 是布尔值，cumsum() 会自动将其视为 1 (True) 和 0 (False)
    # 这里的顺序是按 CSV 中的顺序（即发现顺序）
    cumulative_faults_list = deduplicated_df['is_faulty'].cumsum()
    
    iterations = range(1, len(cumulative_faults_list) + 1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, cumulative_faults_list, label='Cumulative Unique Faults', color='red', linewidth=2)
    plt.fill_between(iterations, cumulative_faults_list, color='red', alpha=0.1)
    
    plt.title('Unique Faults Found vs. Unique Inputs Discovered')
    plt.xlabel('Number of Unique Inputs Discovered')
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
    plt.close()


# --- 4. 图表2：t-SNE 空间图 ---

def run_tsne(data, n_samples, n_dimensions):
    """
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
        max_iter=1000, 
        random_state=42
    )
    tsne_results = tsne.fit_transform(data)
    
    end_time = time.time()
    print(f"t-SNE 运行完成，耗时: {end_time - start_time:.2f} 秒")
    return tsne_results

def plot_full_space(deduplicated_df):
    """
    绘制 t-SNE 结果的 2D 散点图
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
    labels = np.array(labels_list) 
    
    # 运行 t-SNE
    tsne_results = run_tsne(all_data, all_data.shape[0], all_data.shape[1])
    
    # 绘制
    print(f"正在绘制 {tsne_results.shape[0]} 个数据点...")
    faulty_points = tsne_results[labels == True]
    non_faulty_points = tsne_results[labels == False]
    
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
    
    plt.title('t-SNE Visualization of Unique Explored Inputs')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    try:
        plt.savefig(PLOT_2_FILE)
        print(f"[图表 2] 已保存到: {PLOT_2_FILE}")
    except Exception as e:
        print(f"[图表 2] 保存图表时出错: {e}")
    plt.close()


# --- 5. 图表3：故障代数直方图 ---

def plot_generation_histogram(deduplicated_df):
    """
    绘制 *独特故障输入* 的代数直方图。
    """
    print(f"\n[图表 3] 正在分析故障代数...")
    
    # 筛选出所有 'is_faulty' 为 True 的独特输入
    faulty_df = deduplicated_df[deduplicated_df['is_faulty'] == True]
    
    # 获取这些故障的 'mutation_count'
    fault_generations = faulty_df['mutation_count'].values.astype(int) 
            
    print(f"分析完成。总共找到 {len(fault_generations)} 个独特的故障事件。")
    
    if len(fault_generations) == 0:
        print("[图表 3] 未找到故障代数数据，无法绘图。")
        return

    generation_counts = Counter(fault_generations)
    
    max_gen = 0
    if generation_counts:
        max_gen = max(generation_counts.keys())
        
    generations = range(0, max_gen + 2) 
    counts = [generation_counts.get(gen, 0) for gen in generations]
    
    plt.figure(figsize=(12, 7))
    plt.bar(generations, counts, color='red', alpha=0.7, zorder=3)
    
    plt.title('Histogram of Unique Fault Generations')
    plt.xlabel('Mutation Generation (Depth)') 
    plt.ylabel('Number of Unique Faulty Inputs')
    
    step = max(1, (max_gen // 20)) 
    plt.xticks(np.arange(0, max_gen + 2, step=step))
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    try:
        plt.savefig(PLOT_3_FILE)
        print(f"[图表 3] 已保存到: {PLOT_3_FILE}")
    except Exception as e:
        print(f"[图表 3] 保存图表时出错: {e}")
    plt.close()


# --- [新增] 6. 图表4：故障随时间变化图 ---

def plot_faults_over_time(deduplicated_df):
    """
    绘制发现的独特故障数量随时间(小时)的变化曲线。
    """
    print(f"\n[图表 4] 正在计算故障随时间变化趋势...")
    
    # 1. 筛选出故障数据
    faulty_df = deduplicated_df[deduplicated_df['is_faulty'] == True].copy()
    
    if faulty_df.empty:
        print("[图表 4] 未找到故障数据，跳过绘图。")
        return

    # 2. 确保按时间排序 (elapsed_time 应该是秒)
    faulty_df = faulty_df.sort_values(by='elapsed_time')
    
    # 3. 准备数据
    # 将秒转换为小时 (如果您更喜欢分钟，除以 60 即可)
    times_hours = faulty_df['elapsed_time'] / 3600.0
    
    # 累积计数: 第1个故障是1，第2个是2...
    cumulative_counts = range(1, len(faulty_df) + 1)
    
    # 4. 绘图
    plt.figure(figsize=(12, 7))
    
    # 使用 step 图可以更清晰地显示离散的发现过程 ('post' 表示阶梯在点之后变化)
    plt.step(times_hours, cumulative_counts, where='post', color='darkorange', linewidth=2, label='Cumulative Crashes')
    plt.fill_between(times_hours, cumulative_counts, step='post', color='darkorange', alpha=0.1)
    
    plt.title('Cumulative Unique Faults Found vs. Time')
    plt.xlabel('Time Elapsed (Hours)') 
    plt.ylabel('Cumulative Number of Unique Faults')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 设置坐标轴范围
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    
    try:
        plt.savefig(PLOT_4_FILE)
        print(f"[图表 4] 已保存到: {PLOT_4_FILE}")
    except Exception as e:
        print(f"[图表 4] 保存图表时出错: {e}")
    plt.close()


# --- 7. 主函数 ---

def main():
    # 1. 加载并去重数据
    deduplicated_df = load_and_deduplicate(LOG_FILE)
    
    if deduplicated_df is None or deduplicated_df.empty:
        print("未能从日志中提取任何有效数据。退出。")
        return
        
    # 2. 运行图表 1 (按输入数量)
    plot_fault_trend(deduplicated_df)
    
    # 3. 运行图表 2 (t-SNE)
    plot_full_space(deduplicated_df)
    
    # 4. 运行图表 3 (直方图)
    plot_generation_histogram(deduplicated_df)
    
    # 5. [新增] 运行图表 4 (按时间)
    plot_faults_over_time(deduplicated_df)
        
    print("\n所有分析和绘图已完成。")

if __name__ == "__main__":
    main()