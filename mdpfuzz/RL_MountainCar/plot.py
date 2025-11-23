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
LOG_FILE = 'rt_10_0.01_0.01_1022_logs.txt' 
PLOT_1_FILE = 'rt_crashes_over_unique_inputs.png'
PLOT_2_FILE = 'rt_full_input_space_tsne.png'
PLOT_3_FILE = 'rt_crash_generation_histogram.png'

# --- 2. 核心辅助函数 (已修改) ---

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
        print(f"检测到的列名: {df.columns.tolist()}")

        # --- 关键假设 ---
        if df['Oracle'].dtype == 'object':
            print("  'Oracle' 列是 object 类型, 正在映射 'True' -> True, 'False' -> False")
            df['Oracle'] = df['Oracle'].map({'True': True, 'False': False, 'None': None})
        
        # --- *** 已修改：假设 Oracle == True 是崩溃 *** ---
        df['is_crash'] = (df['Oracle'] == True)
        
        # --- 转换 object 类型的列 ---
        # 确保 'Sensitivity', 'Coverage', 'CoverageTime' 是数字类型
        for col in ['Sensitivity', 'Coverage', 'CoverageTime']:
            if col in df.columns and df[col].dtype == 'object':
                print(f"  正在转换 {col} 列: 将 'None' 字符串替换为 NaN 并转换为 float...")
                df[col] = pd.to_numeric(df[col], errors='coerce')

        
        print("\n--- 数据初步检查 ---")
        df.info() # 打印 df.info() 的输出以进行验证
        print(df.head())
        
        crash_counts = df['is_crash'].value_counts()
        
        # --- *** 已修改：更新日志消息 *** ---
        print(f"\n--- 崩溃分析 (基于 'Oracle == True' 假设) ---")
        print(crash_counts)
        
        if True not in crash_counts or crash_counts[True] == 0:
            # --- *** 已修改：更新日志消息 *** ---
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

# --- 3. 绘图函数 (改编自 plotmax.py) ---

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
    
    # 添加版本检查
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
        
        # --- *** 错误修复 (第 3 轮) *** ---
        # 错误 "unexpected keyword argument 'max_iter'" (来自 Run 2)
        # 错误 "unexpected keyword argument 'n_iter'" (来自 Run 1)
        # 这表明 VM 的 sklearn 版本不接受 'n_iter' 或 'max_iter' 作为构造函数参数。
        # 我们将移除此参数，使用默认的迭代次数。
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, init='pca') # <--- 修正行
        
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
        
        # --- *** 新增功能：计算平均值和中位数 *** ---
        if not crash_generations.empty:
            mean_gen = crash_generations.mean()
            median_gen = crash_generations.median()
            print(f"  [统计] 导致独特崩溃的平均代数: {mean_gen:.2f}")
            print(f"  [统计] 导致独特崩溃的代数中位数: {median_gen:.2f}")
        else:
            print("  [统计] 'Generation' 列为空或无效，无法计算统计数据。")
        # --- *** 新增功能结束 *** ---
        
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

    end_time = time.time()
    print(f"\n--- 脚本执行完毕，总耗时: {end_time - start_time:.2f} 秒 ---")

if __name__ == "__main__":
    main()