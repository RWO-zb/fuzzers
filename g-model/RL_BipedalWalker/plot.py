# g-model/RL_BipedalWalker/plot.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from collections import Counter
from sklearn.manifold import TSNE # 依赖于 plot_full_space.py 的风格

# --- 配置 ---
IMG_DIR = 'results' 
LOG_FILE = 'results/all_fuzz_cases_log.pickle'
# --- 结束配置 ---

def load_data(file_path):
    """
    加载 pickle 文件 (仿照: plot.py, findstep.py)
    """
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        print("请先运行 enjoy.py 生成该文件。")
        return None
    
    print(f"正在从 {file_path} 加载数据...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"加载 pickle 文件时出错: {e}")
        return None

def preprocess_data_for_deduplication(data):
    """
    核心函数：对所有迭代数据进行去重。
    我们将遍历所有 N 次迭代，构建一个关于 *独特* 输入状态的字典。
    """
    print("正在对所有日志数据进行去重预处理...")
    
    all_states = data['all_mutate_states'] # (N, 15)
    all_gens = data['all_generations']     # (N,)
    all_crashes = data['all_is_crash']     # (N,)
    
    # unique_states_map 结构:
    # key: state_bytes (独特输入)
    # value: {
    #    'state_array': 15D numpy 数组
    #    'first_seen_iteration': 首次发现此输入的迭代索引 (int)
    #    'first_seen_generation': 首次发现此输入时的代数 (int)
    #    'did_crash': 此输入是否 *曾经* 导致过崩溃 (int, 0 或 1)
    # }
    unique_states_map = {}

    for i in range(len(all_states)):
        state_array = all_states[i]
        state_bytes = state_array.tobytes()
        did_crash = all_crashes[i]
        gen = all_gens[i]
        
        if state_bytes not in unique_states_map:
            # 第一次见到这个独特输入
            unique_states_map[state_bytes] = {
                'state_array': state_array,
                'first_seen_iteration': i,
                'first_seen_generation': gen,
                'did_crash': did_crash
            }
        else:
            # 已经见过这个输入。我们只关心是否需要更新 'did_crash' 状态
            # (仿照 plot_full_space.py 的逻辑)
            unique_states_map[state_bytes]['did_crash'] = max(
                unique_states_map[state_bytes]['did_crash'], did_crash
            )
            
    print(f"预处理完成。总迭代次数 {len(all_states)}，独特输入状态 {len(unique_states_map)}")
    return unique_states_map


def plot_unique_crashes_vs_discovery(unique_states_map):
    """
    1. 绘制 *独特* Crash 数量随 *独特* 输入发现数量的曲线图
       (HACK: 此图现在已完全去重)
    """
    print("\n[1/3] 正在绘制 独特崩溃 vs. 独特输入 (仿照 plot.py)...")
    
    try:
        # 按 "首次发现的迭代" 排序，以重建发现历史
        # (仿照 plot.py 的迭代逻辑)
        unique_entries = sorted(unique_states_map.values(), key=lambda x: x['first_seen_iteration'])
        
        if not unique_entries:
            print("  警告: 未找到独特的崩溃数据。")
            return

        # 提取崩溃状态 (0 或 1)
        crash_status_list = [entry['did_crash'] for entry in unique_entries]
        
        # 计算累积的 *独特* 崩溃
        cumulative_unique_crashes = np.cumsum(crash_status_list)
        
        # X 轴：发现的独特输入的数量
        unique_inputs_discovered = np.arange(1, len(unique_entries) + 1)
        
        plt.figure(figsize=(12, 7))
        # (仿照 plot.py 的绘图风格)
        plt.plot(unique_inputs_discovered, cumulative_unique_crashes, label='Unique Crashes Found', color='red', linewidth=2)
        plt.fill_between(unique_inputs_discovered, cumulative_unique_crashes, color='red', alpha=0.1)
        
        plt.title('Unique Crashes Found vs. Unique Inputs Discovered (De-duplicated)', fontsize=14)
        plt.xlabel('Number of Unique Inputs Discovered', fontsize=12)
        plt.ylabel('Cumulative Number of Unique Crashes', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(0, len(unique_inputs_discovered))
        plt.ylim(bottom=0)
    
        save_path = os.path.join(IMG_DIR, 'crashes_over_time_deduplicated.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  图像已保存到: {save_path}")
        
    except Exception as e:
        print(f"  绘制 独特崩溃 vs. 独特输入 时出错: {e}")


def plot_mutate_state_tsne(unique_states_map):
    """
    2. 绘制 mutate_state (15D 输入) 的 t-SNE 降维二维分布图
       (风格和逻辑借鉴自: plot_full_space.py)
       (HACK: 此函数现在使用预处理的 unique_states_map)
    """
    print("\n[2/3] 正在绘制 mutate_state t-SNE 覆盖率 (仿照 plot_full_space.py)...")
    
    try:
        if not unique_states_map:
            print("  警告: 未找到 'all_mutate_states' 数据。")
            return
            
        # 准备 t-SNE 数据
        state_array = np.array([entry['state_array'] for entry in unique_states_map.values()])
        labels = np.array([entry['did_crash'] for entry in unique_states_map.values()])

        print(f"  总共找到 {state_array.shape[0]} 个独特的输入状态。")

        print(f"  正在运行 t-SNE (n_components=2) ... 这可能需要一些时间 ...")
        start_time = time.time()
        
        # 修复: n_iter -> max_iter
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, max_iter=1000, init='pca', learning_rate='auto')
        
        tsne_data = tsne.fit_transform(state_array)
        print(f"  t-SNE 运行完成，耗时: {time.time() - start_time:.2f} 秒。")

        crashing_points = tsne_data[labels == 1]
        non_crashing_points = tsne_data[labels == 0]
        
        print(f"  非崩溃点: {non_crashing_points.shape[0]}")
        print(f"  崩溃点: {crashing_points.shape[0]}")
        
        plt.figure(figsize=(12, 10))
        
        plt.scatter(
            non_crashing_points[:, 0], 
            non_crashing_points[:, 1], 
            c='blue', 
            alpha=0.4,
            s=10, 
            label=f'Non-Crashing Inputs ({non_crashing_points.shape[0]})'
        )
        
        if crashing_points.shape[0] > 0:
            plt.scatter(
                crashing_points[:, 0], 
                crashing_points[:, 1], 
                c='red', 
                alpha=0.8,
                s=15, 
                label=f'Crashing Inputs ({crashing_points.shape[0]})'
            )
        
        plt.title('t-SNE Visualization of Explored Input Space (15D Ground Types -> 2D)', fontsize=14)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        save_path = os.path.join(IMG_DIR, 'full_input_space_tsne.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  图像已保存到: {save_path}")

    except Exception as e:
        print(f"  绘制 mutate_state t-SNE 时出错: {e}")
        import traceback
        traceback.print_exc()


def plot_crash_generation_histogram(unique_states_map):
    """
    3. 绘制导致崩溃种子的代数的图
       (风格和逻辑借鉴自: findstep.py)
       (HACK: 此函数现在使用预处理的 unique_states_map)
    """
    print("\n[3/3] 正在绘制 崩溃代数直方图 (仿照 findstep.py)...")
    
    try:
        # 提取所有 *独特* 崩溃输入的 *首次发现* 代数
        crashing_generations = [
            entry['first_seen_generation'] 
            for entry in unique_states_map.values() 
            if entry['did_crash'] == 1
        ]
        
        if not crashing_generations:
            print("  警告: 未找到崩溃种子，无法绘制代数图。")
            return
        
        print(f"  总共找到 {len(crashing_generations)} 个 *独特* 的崩溃输入。")

        # 统计 (仿照 findstep.py)
        generation_counts = Counter(crashing_generations)
        
        crashing_gen_array = np.array(crashing_generations)
        max_gen = int(crashing_gen_array.max())
        min_gen = int(crashing_gen_array.min())
        generations = np.arange(min_gen, max_gen + 2) 
        counts = [generation_counts.get(gen, 0) for gen in generations] 
        
        print("\n  --- 独特崩溃代数统计 (已去重) ---")
        print(f"  平均崩溃代数: {crashing_gen_array.mean():.2f}")
        print(f"  中位崩溃代数: {np.median(crashing_gen_array)}")
        print(f"  最小崩溃代数: {min_gen}")
        print(f"  最大崩溃代数: {max_gen}")

        plt.figure(figsize=(12, 7))
        # (仿照 findstep.py)
        plt.bar(generations, counts, color='red', alpha=0.7, align='center', width=0.8) 
        
        plt.title('Histogram of Unique Crash Generations (De-duplicated)', fontsize=14)
        plt.xlabel('Mutation Generation (Depth from Initial Seed)', fontsize=12)
        plt.ylabel('Number of Unique Crashes Found', fontsize=12)
        
        if max_gen - min_gen < 50:
             plt.xticks(np.arange(min_gen, max_gen + 1))
        
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        save_path = os.path.join(IMG_DIR, 'crash_generation_histogram.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  图像已保存到: {save_path}")

    except Exception as e:
        print(f"  绘制 崩溃代数 时出错: {e}")


def main():
    # 使用在脚本顶部定义的硬编码 LOG_FILE
    input_file = LOG_FILE
    
    # 加载数据
    data = load_data(input_file)
    if data is None:
        return

    # 检查数据完整性
    required_keys = ['all_mutate_states', 'all_generations', 'all_is_crash']
    if not all(key in data for key in required_keys):
        print("错误: Pickle 文件中缺少必要的键。")
        print(f"需要: {required_keys}")
        print(f"已有: {list(data.keys())}")
        return

    # 创建输出目录
    os.makedirs(IMG_DIR, exist_ok=True) 

    # -----------------------------------------------------------------
    # HACK: 在所有绘图之前执行核心去重步骤
    # -----------------------------------------------------------------
    unique_states_map = preprocess_data_for_deduplication(data)

    # 运行所有绘图函数
    # 1. 崩溃 vs 独特输入
    plot_unique_crashes_vs_discovery(unique_states_map)
    # 2. t-SNE 覆盖率
    plot_mutate_state_tsne(unique_states_map)
    # 3. 崩溃代数
    plot_crash_generation_histogram(unique_states_map)
    
    print(f"\n所有绘图已完成并保存到 '{IMG_DIR}' 文件夹中。")

if __name__ == '__main__':
    main()