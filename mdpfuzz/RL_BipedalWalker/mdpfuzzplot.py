import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn # 导入 sklearn 以检查版本
from sklearn.manifold import TSNE
from collections import Counter
import os
import time
import ast # 用于安全地将字符串 '[1,2,3]' 转换为列表 [1,2,3]

# --- 1. 配置 ---
LOG_FILE = 'rt_10_0.01_0.01_0_logs.txt' 
PLOT_1_FILE = 'rt_crashes_over_unique_inputs.png'
PLOT_2_FILE = 'rt_full_input_space_tsne.png'
PLOT_3_FILE = 'rt_crash_generation_histogram.png'
PLOT_4_FILE = 'rt_crashes_over_time.png' # <--- 新增：图表 4 的文件名

# --- 2. 核心辅助函数 ---

def load_and_prepare_data(file_path):
    """
    加载 fuzzer_..._logs.txt 文件, 并进行预处理
    """
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        return None
    
    try:
        print(f"正在从 {file_path} 加载原始日志数据...")
        
        df = pd.read_csv(
            file_path, 
            delimiter=';', 
            on_bad_lines='skip', 
            skipinitialspace=True 
        )
        
        print(f"原始日志加载完成，总共 {len(df)} 条记录。")
        # print(f"检测到的列名: {df.columns.tolist()}")

        # --- 关键假设 ---
        if df['Oracle'].dtype == 'object':
            # print("  'Oracle' 列是 object 类型, 正在映射 'True' -> True, 'False' -> False")
            df['Oracle'] = df['Oracle'].map({'True': True, 'False': False, 'None': None})
        
        # --- 假设 Oracle == True 是崩溃 ---
        df['is_crash'] = (df['Oracle'] == True)
        
        # --- 转换 object 类型的列 ---
        # 确保 'Sensitivity', 'Coverage', 'CoverageTime', 'RunTime' 是数字类型
        # 新增 'RunTime' 以支持基于时间的绘图
        for col in ['Sensitivity', 'Coverage', 'CoverageTime', 'RunTime']:
            if col in df.columns:
                # print(f"  正在处理 {col} 列: 确保为数值类型...")
                df[col] = pd.to_numeric(df[col], errors='coerce')

        
        print("\n--- 数据初步检查 ---")
        # df.info() 
        print(df.head())
        
        crash_counts = df['is_crash'].value_counts()
        
        print(f"\n--- 崩溃分析 (基于 'Oracle == True' 假设) ---")
        print(crash_counts)
        
        if True not in crash_counts or crash_counts[True] == 0:
            print("警告: 根据当前假设 (Oracle == True)，未找到任何崩溃。")
        else:
            print(f"根据假设，共找到 {crash_counts[True]} 条崩溃记录。")
        
        return df
        
    except KeyError as e:
        print(f"加载或处理数据时发生 KeyError: {e}")
        try:
            temp_df = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip', skipinitialspace=True, nrows=1)
            print(f"Pandas 读取到的列名是: {temp_df.columns.tolist()}")
        except Exception as read_e:
            print(f"读取列名失败: {read_e}")
        return None
    except Exception as e:
        print(f"加载或处理 CSV 文件时出错: {e}")
        return None

def deduplicate_log(original_data_df):
    """
    根据 'Input' 列对日志进行去重
    """
    if original_data_df is None:
        return None
        
    print(f"\n正在对 {len(original_data_df)} 条记录进行去重 (基于 'Input' 列)...")
    try:
        # keep='first' 保留第一次出现的记录，这对于基于时间的分析很重要
        unique_df = original_data_df.drop_duplicates(subset=['Input'], keep='first')
        unique_df = unique_df.reset_index(drop=True) # 重置索引
        print(f"去重后，剩余 {len(unique_df)} 条独特记录。")
        return unique_df
    except KeyError:
        print("错误: 数据中未找到 'Input' 列。")
        return None
    except Exception as e:
        print(f"去重时出错: {e}")
        return None

# --- 3. 绘图函数 ---

def plot_crashes_over_time(unique_log_df):
    """
    图 1: 绘制独特崩溃数随独特输入数量的变化 (累计)
    """
    if unique_log_df is None:
        print("[图表 1] 跳过：无数据。")
        return

    print("\n[图表 1] 正在生成 '独特崩溃 vs 独特输入' 图...")
    cumulative_crashes = unique_log_df['is_crash'].cumsum()
    
    plt.figure(figsize=(12, 7))
    plt.plot(cumulative_crashes)
    plt.title('Total Unique Crashes Found Over Time (vs. Unique Inputs Seen)')
    plt.xlabel('Number of Unique Inputs Explored')
    plt.ylabel('Cumulative Number of Unique Crashes')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    try:
        plt.savefig(PLOT_1_FILE)
        print(f"[图表 1] 已保存到: {PLOT_1_FILE}")
    except Exception as e:
        print(f"[图表 1] 保存图表时出错: {e}")
    plt.close()


def plot_tsne_all(unique_log_df):
    """
    图 2: 绘制所有独特输入的 t-SNE 降维图，并高亮崩溃输入
    """
    if unique_log_df is None:
        print("[图表 2] 跳过：无数据。")
        return
        
    print("\n[图表 2] 正在生成 't-SNE 输入空间' 图...")
    print(f"  使用 scikit-learn version: {sklearn.__version__}")

    try:
        # 1. 解析 'Input' 字符串
        print("  正在解析 'Input' 列表...")
        inputs_list = unique_log_df['Input'].apply(ast.literal_eval)
        
        # 2. 转换为 Numpy 数组
        X = np.array(inputs_list.tolist())
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        print(f"  t-SNE 数据塑形为: {X.shape}")
        
        # 3. 运行 t-SNE
        n_samples = X.shape[0]
        if n_samples <= 1:
            print("[图表 2] t-SNE 失败：数据点不足 (<= 1)。")
            return
            
        perplexity_value = min(30, n_samples - 1)
        if perplexity_value <= 0:
             print("[图表 2] t-SNE 失败：无法设置有效的 perplexity。")
             return
             
        print(f"  正在运行 t-SNE (n_samples={n_samples}, perplexity={perplexity_value})...")
        
        # 兼容不同版本的 sklearn，不传入 max_iter 或 n_iter
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, init='pca')
        
        X_2d = tsne.fit_transform(X)
        print("  t-SNE 完成。")

        # 4. 准备绘图数据
        plot_df = pd.DataFrame(X_2d, columns=['x', 'y'])
        plot_df['is_crash'] = unique_log_df['is_crash']
        
        non_crashes = plot_df[plot_df['is_crash'] == False]
        crashes = plot_df[plot_df['is_crash'] == True]

        # 5. 绘图
        plt.figure(figsize=(12, 10))
        plt.scatter(non_crashes['x'], non_crashes['y'], c='blue', alpha=0.3, label='Non-Crashing Inputs', s=10)
        
        if not crashes.empty:
            plt.scatter(crashes['x'], crashes['y'], c='red', alpha=1.0, label='Crashing Inputs', s=50, edgecolors='black')
        
        plt.title('t-SNE Visualization of Unique Input Space')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        try:
            plt.savefig(PLOT_2_FILE)
            print(f"[图表 2] 已保存到: {PLOT_2_FILE}")
        except Exception as e:
            print(f"[图表 2] 保存图表时出错: {e}")
        plt.close()

    except Exception as e:
        print(f"[图表 2] 生成 t-SNE 图时出错: {e}")
        print("  t-SNE 失败。可能是 'Input' 列格式不统一或数据点太少。")


def plot_crash_generation_histogram(unique_log_df):
    """
    图 3: 绘制首次发现崩溃的 "Generation" (代数) 的直方图
    """
    if unique_log_df is None:
        print("[图表 3] 跳过：无数据。")
        return

    print("\n[图表 3] 正在生成 '崩溃代数直方图'...")
    crash_data = unique_log_df[unique_log_df['is_crash'] == True]
    
    if crash_data.empty:
        print("[图表 3] 未找到独特的崩溃记录，跳过绘图。")
        return

    try:
        crash_generations = crash_data['Generation'].astype(int)
        
        if not crash_generations.empty:
            mean_gen = crash_generations.mean()
            median_gen = crash_generations.median()
            print(f"  [统计] 导致独特崩溃的平均代数: {mean_gen:.2f}")
            print(f"  [统计] 导致独特崩溃的代数中位数: {median_gen:.2f}")
        
        generation_counts = Counter(crash_generations)
        
        if not generation_counts:
            print("[图表 3] 崩溃数据中未找到 'Generation' 信息。")
            return

        generations = sorted(generation_counts.keys())
        counts = [generation_counts[g] for g in generations]
        max_gen = max(generations) if generations else 1
        
        print(f"  独特崩溃代数范围: 0 - {max_gen}")
        print(f"  总独特崩溃数: {sum(counts)}")

        plt.figure(figsize=(12, 7))
        plt.bar(generations, counts, color='red', alpha=0.7, zorder=3)
        plt.title('Histogram of Unique Crash Generations (by First Appearance)')
        plt.xlabel('Mutation Generation (Depth from Initial Seed)')
        plt.ylabel('Number of Unique Crashing Inputs')
        
        step = max(1, (max_gen // 20)) 
        if max_gen > 0:
            plt.xticks(np.arange(0, max_gen + 2, step=step))
        plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
        
        try:
            plt.savefig(PLOT_3_FILE)
            print(f"[图表 3] 已保存到: {PLOT_3_FILE}")
        except Exception as e:
            print(f"[图表 3] 保存图表时出错: {e}")
        plt.close()
        
    except KeyError:
        print("[图表 3] 错误: 数据中未找到 'Generation' 列。")
    except Exception as e:
        print(f"[图表 3] 绘制直方图时出错: {e}")

def plot_crashes_over_wallclock_time(unique_log_df):
    """
    图 4: 绘制独特崩溃随时间 (小时) 的变化曲线 (仿照 curefuzzplot.py)
    """
    if unique_log_df is None:
        print("[图表 4] 跳过：无数据。")
        return

    print("\n[图表 4] 正在生成 '独特崩溃 vs 时间' 图...")
    
    # 检查是否有 'RunTime' 列
    if 'RunTime' not in unique_log_df.columns:
        print("[图表 4] 错误: 数据中未找到 'RunTime' 列，无法计算时间。")
        return

    # 提取崩溃数据
    crash_df = unique_log_df[unique_log_df['is_crash'] == True].copy()
    
    if crash_df.empty:
        print("[图表 4] 未找到独特的崩溃记录，跳过绘图。")
        return

    # 计算相对时间 (小时)
    # 我们使用日志中的最小时间戳作为实验开始时间
    # 注意：为了准确性，最好是使用整个日志的开始时间，但这里我们使用去重后数据的最早时间作为近似
    # 如果数据是按时间顺序记录的，这通常是准确的。
    start_time = unique_log_df['RunTime'].min()
    
    # 计算崩溃发生时相对于开始时间的秒数
    crash_times = crash_df['RunTime'] - start_time
    
    # 转换为小时
    crash_times_hours = crash_times / 3600.0
    
    # 排序 (确保时间单调递增)
    crash_times_hours = crash_times_hours.sort_values()
    
    # 累积计数 (1, 2, 3...)
    cumulative_counts = np.arange(1, len(crash_times_hours) + 1)
    
    print(f"  共找到 {len(crash_times_hours)} 个独特崩溃。")
    if not crash_times_hours.empty:
        print(f"  首次崩溃时间: {crash_times_hours.iloc[0]:.4f} 小时")
        print(f"  最后崩溃时间: {crash_times_hours.iloc[-1]:.4f} 小时")

    plt.figure(figsize=(12, 7))
    
    # 使用 step 图可以更清晰地显示离散的发现过程
    plt.step(crash_times_hours, cumulative_counts, where='post', color='darkorange', linewidth=2, label='Cumulative Crashes')
    plt.fill_between(crash_times_hours, cumulative_counts, step='post', color='darkorange', alpha=0.1)
    
    plt.title('Cumulative Unique Crashes vs. Time')
    plt.xlabel('Time Elapsed (hours)')
    plt.ylabel('Cumulative Number of Unique Crashes')
    
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

# --- 4. 主函数 ---

def main():
    print("--- Fuzzer 日志分析脚本 (改编自 plotmax.py) ---")
    start_time = time.time()
    
    original_log_data_df = load_and_prepare_data(LOG_FILE)
    
    if original_log_data_df is None:
        print("数据加载失败，脚本终止。")
        return

    unique_log_df = deduplicate_log(original_log_data_df)
    
    if unique_log_df is None:
        print("数据去重失败，脚本终止。")
        return

    # --- 3. 调用绘图函数 ---
    plot_crashes_over_time(unique_log_df)
    plot_tsne_all(unique_log_df)
    plot_crash_generation_histogram(unique_log_df)
    plot_crashes_over_wallclock_time(unique_log_df) # <--- 新增调用

    end_time = time.time()
    print(f"\n--- 脚本执行完毕，总耗时: {end_time - start_time:.2f} 秒 ---")

if __name__ == "__main__":
    main()