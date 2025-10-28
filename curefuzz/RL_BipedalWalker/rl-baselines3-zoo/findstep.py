import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

# --- 配置 ---
LOG_FILE = 'selection_log.pkl'
PLOT_FILE_NAME = 'crash_generation_histogram.png'
# --- 结束配置 ---

def load_data(file_path):
    """加载 selection_log.pkl 文件"""
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"加载 pickle 文件时出错: {e}")
        return None

def build_parent_map(log_data):
    """
    遍历日志，构建一个 {seed_id: parent_id} 的家谱图。
    """
    parent_map = {}
    for entry in log_data:
        seed_id = entry.get('seed_id')
        parent_id = entry.get('parent_id')
        
        if seed_id is not None:
            # 我们只关心每个
            if seed_id not in parent_map:
                 parent_map[seed_id] = parent_id
    
    print(f"家谱图构建完成，总共找到 {len(parent_map)} 个独特的父种子。")
    return parent_map

# 这是一个“记忆化”（memoization）字典，用于缓存已计算过的代数，避免重复计算
generation_cache = {}

def get_generation(seed_id, parent_map):
    """
    递归函数：追溯一个 seed_id 的代数（即变异深度）。
    """
    # 1. 如果已经计算过，直接返回缓存的结果
    if seed_id in generation_cache:
        return generation_cache[seed_id]
        
    # 2. 检查 parent_id 是否存在
    if seed_id not in parent_map:
        # 理论上不应该发生，除非是数据丢失或根节点
        parent_id = None
    else:
        parent_id = parent_map[seed_id]

    # 3. 基础情况：如果 parent_id 是 None，说明这是第 0 代
    if parent_id is None:
        generation_cache[seed_id] = 0
        return 0
        
    # 4. 递归：代数 = 1 + (父代的代数)
    try:
        parent_generation = get_generation(parent_id, parent_map)
        current_generation = parent_generation + 1
        
        # 缓存结果
        generation_cache[seed_id] = current_generation
        return current_generation
        
    except RecursionError:
        print(f"错误：在家谱中检测到循环！ seed_id: {seed_id}, parent_id: {parent_id}")
        return -1 # 返回错误代码
    except KeyError:
        print(f"错误：在家谱中找不到 parent_id: {parent_id} (来自 seed_id: {seed_id})")
        return -1 # 返回错误代码


def analyze_crashes(log_data, parent_map):
    """
    遍历日志，找到所有崩溃，并计算它们的代数。
    """
    crash_generations = []
    
    print("正在分析所有崩溃事件...")
    for i, entry in enumerate(log_data):
        if entry.get('did_crash', False):
            # 这是一个崩溃事件。
            # 找到导致这次变异的父种子 ID
            parent_seed_id = entry.get('seed_id')
            
            if parent_seed_id is None:
                continue
            
            # 1. 计算父种子的代数
            parent_generation = get_generation(parent_seed_id, parent_map)
            
            if parent_generation == -1: # 跳过错误
                continue
                
            # 2. 崩溃的种子(mutate_state)是父种子的下一代
            crash_generation = parent_generation + 1
            
            crash_generations.append(crash_generation)
            
    print(f"分析完成。总共记录 {len(crash_generations)} 次崩溃事件。")
    return crash_generations

def plot_histogram(generation_data):
    """
    绘制崩溃代数的直方图。
    """
    if not generation_data:
        print("未找到崩溃代数数据，无法绘图。")
        return

    # 统计每个代数的崩溃次数
    generation_counts = Counter(generation_data)
    
    # 准备绘图数据
    max_gen = max(generation_counts.keys())
    generations = range(0, max_gen + 2) # X轴
    counts = [generation_counts.get(gen, 0) for gen in generations] # Y轴
    
    # 打印统计数据
    print("\n--- 崩溃代数统计 ---")
    print(f"  平均崩溃代数: {np.mean(generation_data):.2f}")
    print(f"  中位崩溃代数: {np.median(generation_data)}")
    print(f"  最小崩溃代数: {np.min(generation_data)}")
    print(f"  最大崩溃代数: {np.max(generation_data)}")
    print("  按代数分布:")
    for gen, count in sorted(generation_counts.items()):
        print(f"    第 {gen} 代: {count} 次崩溃")

    # 绘制条形图
    plt.figure(figsize=(12, 7))
    plt.bar(generations, counts, color='red', alpha=0.7)
    
    plt.title('Histogram of Crash Generations')
    plt.xlabel('Mutation Generation (Depth from Initial Seed)')
    plt.ylabel('Number of Crashes Found')
    plt.xticks(generations) # 确保 X 轴显示所有整数
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 保存图表
    try:
        plt.savefig(PLOT_FILE_NAME)
        print(f"\n图表已保存到: {PLOT_FILE_NAME}")
    except Exception as e:
        print(f"保存图表时出错: {e}")

def main():
    log_data = load_data(LOG_FILE)
    if log_data:
        parent_map = build_parent_map(log_data)
        crash_generations = analyze_crashes(log_data, parent_map)
        plot_histogram(crash_generations)

if __name__ == "__main__":
    main()