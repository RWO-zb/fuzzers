import matplotlib.pyplot as plt
import re
import pandas as pd

def plot_crashes_vs_samples(log_filename='D:\\code\\fuzzers\\mdpfuzz\\data_rq2\\bw\\fuzzer_10_0.01_0.01_2021_logs.txt', output_filename='D:\\code\\fuzzers\\mdpfuzz\\data_rq2\\bw\\oracle_false_vs_unique_samples.png'):
    """
    读取 fuzzer 日志文件，提取 *唯一的* 样本数据，并绘制累计崩溃与 *唯一* 样本总数的关系图。

    Args:
        log_filename (str): The name of the input log file.
        output_filename (str): The name of the output plot image file.
    """
    total_samples = 0  # 这将是 *唯一* 样本的总数
    crash_count = 0
    samples_list = []
    crashes_list = []
    
    # --- 新增：用于跟踪已见输入的集合 ---
    seen_inputs = set()

    # Regex to find lines representing a sample result
    sample_pattern = re.compile(r"\];\s*(True|False);")

    try:
        with open(log_filename, 'r', encoding='utf-8') as f:
            for line in f:
                match = sample_pattern.search(line)
                if match:
                    # --- 修改：提取输入代表（匹配开始前的所有内容） ---
                    # 我们假设这部分内容可以唯一标识一个输入
                    input_repr = line[:match.start()].strip()
                    
                    # --- 修改：检查是否为重复输入 ---
                    if input_repr in seen_inputs:
                        continue  # 如果是重复的，跳过此行
                    
                    # --- 修改：如果是新输入，添加到集合并处理 ---
                    seen_inputs.add(input_repr)
                    
                    # 只有唯一的样本才会被计数
                    total_samples += 1
                    oracle_value_str = match.group(1)
                    if oracle_value_str == 'False':
                        crash_count += 1

                    # 存储数据点（基于 *唯一* 样本数）
                    samples_list.append(total_samples)
                    crashes_list.append(crash_count)

        if not samples_list:
            # --- 修改：更新提示信息 ---
            print(f"未能从文件 '{log_filename}' 提取到任何 *唯一* 样本数据。")
            return

        # --- 修改：更新提示信息 ---
        print(f"数据提取完成。总 *唯一* 样本数: {total_samples}, 总崩溃数 (来自唯一样本): {crash_count}")

        # --- Plotting ---
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            print("尝试设置 SimHei 字体成功。")
        except Exception as e:
            print(f"无法设置 SimHei 字体: {e}。将使用默认字体。")

        plt.figure(figsize=(12, 7))
        plt.plot(samples_list, crashes_list, marker='.', linestyle='-', markersize=3)

        # --- 修改：更新图表标题和标签以反映“唯一” ---
        plt.title('累计崩溃数量随 *唯一* 总样本数的变化 (Cumulative Crashes vs. Total *Unique* Samples Tested)')
        plt.xlabel('总 *唯一* 样本数量 (Total *Unique* Samples Tested)')
        plt.ylabel('累计发现的崩溃数量 (Cumulative Crashes Found)')
        plt.grid(True)
        plt.tight_layout()

        # --- 修改：建议使用新的输出文件名 ---
        # 原文件名: 'oracle_false_vs_samples.png'
        # 新文件名: 'oracle_false_vs_unique_samples.png'
        plt.savefig(output_filename)
        print(f"图表已保存为: {output_filename}")

    except FileNotFoundError:
        print(f"错误：文件 '{log_filename}' 未找到。")
    except Exception as e:
        print(f"处理文件或绘图时发生错误: {e}")

# --- Function Call ---
# 注意：我修改了默认的输出文件名以反映数据的“唯一性”
plot_crashes_vs_samples(output_filename='D:\\code\\fuzzers\\mdpfuzz\\data_rq2\\bw\\oracle_false_vs_unique_samples.png')