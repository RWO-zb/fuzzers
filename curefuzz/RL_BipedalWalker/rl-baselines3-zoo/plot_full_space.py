import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import time

# --- 配置 ---
LOG_FILE = 'selection_log.pkl'
PLOT_FILE_NAME = 'full_input_space_tsne.png'
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

def extract_unique_states(log_data):
    """
    提取所有独特的 mutate_state，并标记它们是否导致过崩溃。
    如果一个 state 至少导致过一次崩溃，它就被标记为 'crashing'。
    """
    print("正在处理日志数据以查找所有独特的输入...")
    
    unique_states = {} # key = state.tobytes(), value = did_crash (bool)

    for i, entry in enumerate(log_data):
        if entry['mutate_state'] is None:
            continue
            
        try:
            state_bytes = entry['mutate_state'].tobytes()
            did_crash = entry['did_crash']
            
            if state_bytes not in unique_states:
                unique_states[state_bytes] = did_crash
            else:
                if did_crash and not unique_states[state_bytes]:
                    unique_states[state_bytes] = True
                    
        except Exception as e:
            print(f"处理条目 {i} (seed_id {entry.get('seed_id')}) 时出错: {e}")

    if not unique_states:
        print("未找到任何 mutate_state 数据。")
        return None, None

    print(f"总共找到 {len(unique_states)} 个独特的输入。")

    all_data_list = []
    labels_list = []
    
    # 动态检测 dtype，而不是硬编码
    int32_size = 15 * np.dtype(np.int32).itemsize # 15 * 4 = 60
    int64_size = 15 * np.dtype(np.int64).itemsize # 15 * 8 = 120

    try:
        first_key = next(iter(unique_states))
    except StopIteration:
        print("字典为空，无法检测 dtype。")
        return None, None
        
    if len(first_key) == int32_size:
        print("检测到数据类型为 np.int32 (4 字节)")
        dtype_to_use = np.int32
        expected_size = int32_size
    elif len(first_key) == int64_size:
        print("检测到数据类型为 np.int64 (8 字节)")
        dtype_to_use = np.int64
        expected_size = int64_size
    else:
        print(f"错误: 无法识别的字节大小: {len(first_key)} 字节。期望 {int32_size} 或 {int64_size}。")
        return None, None

    for state_bytes, did_crash in unique_states.items():
        if len(state_bytes) != expected_size:
            print(f"警告: 发现一个大小异常的 state buffer (大小: {len(state_bytes)} 字节)，跳过。")
            continue
            
        all_data_list.append(np.frombuffer(state_bytes, dtype=dtype_to_use))
        labels_list.append(1 if did_crash else 0) # 1 = Crash, 0 = No Crash
        
    if not all_data_list:
        print("没有找到有效数据。")
        return None, None

    all_data = np.vstack(all_data_list)
    labels = np.array(labels_list)
    
    return all_data, labels

def run_tsne(data):
    """
    对 (N, 15) 的数据运行 t-SNE 降维到 2 维
    """
    if data.shape[0] < 50:
        print(f"数据点太少 ({data.shape[0]}个)，t-SNE 可能效果不佳。")
        perplexity_value = max(5, data.shape[0] - 1)
    else:
        # Perplexity 默认值为 30，但它必须小于样本数
        perplexity_value = min(30, data.shape[0] - 1)
        
    print(f"正在对 {data.shape[0]} 个独特的输入（15维）运行 t-SNE (Perplexity={perplexity_value})...")
    
    start_time = time.time()
    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=perplexity_value,
        # --- 修复：将 'n_iter' 替换为 'max_iter' ---
        max_iter=1000,         # <--- 这里是修复
        # --- 结束修复 ---
        random_state=42
    )
    tsne_results = tsne.fit_transform(data)
    
    end_time = time.time()
    print(f"t-SNE 运行完成，耗时: {end_time - start_time:.2f} 秒")
    
    return tsne_results # (N, 2) 数组

def plot_full_space(tsne_data, labels):
    """
    绘制 t-SNE 结果的 2D 散点图，用两种颜色区分
    """
    print(f"正在绘制 {tsne_data.shape[0]} 个数据点...")
    
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
    
    plt.title('t-SNE Visualization of Explored Input Space (15D Ground Types -> 2D)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    try:
        plt.savefig(PLOT_FILE_NAME)
        print(f"图表已保存到: {PLOT_FILE_NAME}")
    except Exception as e:
        print(f"保存图表时出错: {e}")

def main():
    log_data = load_data(LOG_FILE)
    if log_data:
        all_data, labels = extract_unique_states(log_data)
        if all_data is not None and labels is not None:
            tsne_results = run_tsne(all_data)
            plot_full_space(tsne_results, labels)
        else:
            print("没有找到可分析的数据。")

if __name__ == "__main__":
    main()