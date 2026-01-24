import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置常量 ---
LOG_FILE = 'selection_log.pkl'
PLOT_BD_CURVE = 'behaviour_diversity_curve.png'      # 行为多样性 (基于网格覆盖)
PLOT_FD_CURVE = 'fault_diversity_curve.png'          # 故障多样性 (基于Crash网格)
PLOT_SC_CURVE = 'state_coverage_curve.png'           # 状态覆盖率 (Unique Mutate States)
PLOT_IS_CURVE = 'initial_seed_crash_curve.png'       # [新增] 初始种子多样性 (Unique Root Seeds causing Crash)

GRID_SIZE = (50, 50)

def load_data(file_path):
    """加载 Pickle 日志数据"""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Loaded {len(data)} entries from {file_path}")
            return data
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None
    
def get_bin_index(value, min_val, max_val, grid_size_dim):
    """计算值在网格中的索引"""
    if max_val <= min_val:
        return 0
    idx = int((value - min_val) / (max_val - min_val) * grid_size_dim)
    return min(max(idx, 0), grid_size_dim - 1)

def calculate_cumulative_trends(log_data):
    """计算各项指标的累积趋势"""
    
    # 1. 预扫描：确定 BD 计算所需的全局范围 (Min/Max)
    all_dists = []
    all_angles = []
    
    for entry in log_data:
        d = entry.get('bd_distance')
        a = entry.get('bd_mean_angle')
        if d is not None and a is not None:
            all_dists.append(d)
            all_angles.append(a)

    if not all_dists: 
        print("Warning: No behavior data found.")
        return None

    min_dist, max_dist = min(all_dists), max(all_dists) + 1e-5
    min_angle, max_angle = min(all_angles), max(all_angles) + 1e-5
    
    print(f"Global Range - Dist: [{min_dist:.2f}, {max_dist:.2f}], Angle: [{min_angle:.2f}, {max_angle:.2f}]")

    # 2. 初始化集合用于去重
    bd_filled_bins = set()     # 行为覆盖
    fd_crash_bins = set()      # 故障覆盖 (网格)
    unique_states = set()      # 状态覆盖 (Mutate State)
    unique_crash_roots = set() # [新增] 初始种子覆盖 (Root Seed)
    
    # 3. 初始化趋势列表
    bd_trend = []
    fd_trend = []
    sc_trend = []
    is_trend = [] # [新增] Initial Seed Trend
    
    # 4. 遍历日志，逐步计算
    for i, entry in enumerate(log_data):
        # --- A. State Coverage (Unique Mutate States) ---
        state = entry.get('mutate_state')
        if state is not None:  
            # 兼容 numpy 数组或 bytes
            state_bytes = state.tobytes() if hasattr(state, 'tobytes') else np.array(state).tobytes()
            unique_states.add(state_bytes)
        sc_trend.append(len(unique_states))

        # --- B. Initial Seed Diversity (Unique Root Seeds causing Crash) ---
        is_crash = entry.get('did_crash', False)
        
        if is_crash:
            # 只有发生 Crash 时才统计 Root Seed
            root_seed = entry.get('root_seed')
            if root_seed is not None:
                # 同样需要转为 bytes 才能存入 set
                root_bytes = root_seed.tobytes() if hasattr(root_seed, 'tobytes') else np.array(root_seed).tobytes()
                unique_crash_roots.add(root_bytes)
        
        is_trend.append(len(unique_crash_roots))

        # --- C. Behaviour & Fault Diversity (Grid Based) ---
        d = entry.get('bd_distance')
        a = entry.get('bd_mean_angle')
        
        if d is not None and a is not None:
            d_idx = get_bin_index(d, min_dist, max_dist, GRID_SIZE[0])
            a_idx = get_bin_index(a, min_angle, max_angle, GRID_SIZE[1])
            bin_loc = (d_idx, a_idx)
            
            bd_filled_bins.add(bin_loc)
        
            if is_crash:
                fd_crash_bins.add(bin_loc)
        
        bd_trend.append(len(bd_filled_bins))
        fd_trend.append(len(fd_crash_bins))
        
    return {
        'x_axis': range(1, len(log_data) + 1), 
        'bd_trend': bd_trend,
        'fd_trend': fd_trend,
        'sc_trend': sc_trend,
        'is_trend': is_trend  # [新增]
    }

def plot_curve(x, y, title, ylabel, filename, color):
    """通用绘图函数"""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2, color=color)
    plt.fill_between(x, y, color=color, alpha=0.1) 
    
    plt.title(title, fontsize=14)
    plt.xlabel('Number of Test Cases', fontsize=12) 
    plt.ylabel(ylabel, fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0, right=len(x))
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def main():
    log_data = load_data(LOG_FILE)
    if not log_data:
        return

    trends = calculate_cumulative_trends(log_data)
    if not trends:
        return
        
    x = trends['x_axis']

    # 1. Behaviour Diversity
    plot_curve(
        x, trends['bd_trend'], 
        title='Behaviour Diversity Growth', 
        ylabel='Cumulative Covered Bins (Behaviour)', 
        filename=PLOT_BD_CURVE,
        color='teal'
    )
    
    # 2. Fault Diversity
    plot_curve(
        x, trends['fd_trend'], 
        title='Fault Diversity Growth', 
        ylabel='Cumulative Covered Crash Bins', 
        filename=PLOT_FD_CURVE,
        color='crimson'
    )
    
    # 3. State Coverage
    plot_curve(
        x, trends['sc_trend'], 
        title='State Coverage Growth', 
        ylabel='Cumulative Unique Mutate States', 
        filename=PLOT_SC_CURVE,
        color='royalblue'
    )

    # 4. [新增] Initial Seed Crash Diversity
    plot_curve(
        x, trends['is_trend'],
        title='Initial Seed Crash Diversity Growth',
        ylabel='Cumulative Unique Initial Seeds causing Crash',
        filename=PLOT_IS_CURVE,
        color='darkorange'
    )

    print("\nAll curves generated successfully.")

if __name__ == "__main__":
    main()