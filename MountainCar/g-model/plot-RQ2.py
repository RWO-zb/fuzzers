import os
import glob
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 配置 ---
PLOT_FILE = 'gmodel_diversity_metrics.png'

# --- 绘图样式设置 (保持与 CureFuzz 一致) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2
})

def get_latest_result_dir(base_path='results'):
    """自动查找 results 目录下最新的 MC_* 实验文件夹"""
    if not os.path.exists(base_path):
        return None
    # 过滤出 g-model 的结果文件夹 (通常以 MC_ 开头)
    dirs = [d for d in glob.glob(os.path.join(base_path, 'MC_*')) if os.path.isdir(d)]
    if not dirs:
        return None
    # 按修改时间排序，取最新的
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir

def load_gmodel_data(result_dir):
    """加载 g-model 生成的 pickle 文件"""
    traj_file = os.path.join(result_dir, 'all_trajectories.pkl')
    log_file = os.path.join(result_dir, 'all_test_cases_log.pkl')
    
    if not os.path.exists(traj_file) or not os.path.exists(log_file):
        print(f"Error: Data files not found in {result_dir}")
        print(f"Looking for:\n - {traj_file}\n - {log_file}")
        return None, None
    
    print(f"Loading trajectories from: {traj_file}")
    print(f"Loading logs from: {log_file}")
    
    try:
        with open(traj_file, 'rb') as f:
            raw_trajectories = pickle.load(f)
        
        with open(log_file, 'rb') as f:
            raw_logs = pickle.load(f)
            
    except Exception as e:
        print(f"Error loading pickle files: {e}")
        return None, None

    # 数据预处理
    obs_seqs = []
    logs = []
    
    # 确保轨迹和日志长度一致（取最小值以防万一）
    min_len = min(len(raw_trajectories), len(raw_logs))
    
    for i in range(min_len):
        # 处理轨迹: 将 list of arrays 转换为单个 numpy array
        seq = raw_trajectories[i]
        seq_arr = np.array(seq)
        obs_seqs.append(seq_arr)
        
        # 处理日志: 提取 is_crash
        entry = raw_logs[i]
        # g-model 的日志里直接有 'is_crash' 字段
        is_crash = entry.get('is_crash', False)
        logs.append({'did_crash': is_crash})

    print(f"Successfully loaded {len(obs_seqs)} episodes.")
    return obs_seqs, logs

class DiversityAnalyzer:
    def __init__(self, obs_seqs, logs, 
                 state_grid_size=(50, 50), 
                 behavior_grid_size=(50, 50)):
        
        self.obs_seqs = obs_seqs
        self.logs = logs
        self.state_grid_size = state_grid_size
        self.behavior_grid_size = behavior_grid_size
        
        # MountainCar 标准状态范围
        self.state_pos_range = (-1.2, 0.6)
        self.state_vel_range = (-0.07, 0.07)
        
        # 行为描述符范围 (参考 CureFuzz 设定)
        self.bd_max_pos_range = (-1.2, 0.6)
        self.bd_avg_speed_range = (0.0, 0.05) 

    def _get_grid_index(self, values, ranges, grid_size):
        indices = []
        for val, (min_val, max_val), bins in zip(values, ranges, grid_size):
            norm = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
            idx = int(norm * bins)
            idx = np.clip(idx, 0, bins - 1)
            indices.append(idx)
        return tuple(indices)

    def _calc_behavior_descriptor(self, sequence):
        seq_arr = np.array(sequence)
        if len(seq_arr) == 0:
            return -1.2, 0.0
        # 假设 sequence 是 (N, 2) 的数组，列 0 是 pos，列 1 是 vel
        if seq_arr.ndim == 1: # 处理可能的一维情况
             seq_arr = seq_arr.reshape(-1, 2)
             
        positions = seq_arr[:, 0]
        velocities = seq_arr[:, 1]
        
        max_pos = np.max(positions)
        avg_speed = np.mean(np.abs(velocities))
        return max_pos, avg_speed

    def calculate_trends(self):
        if not self.obs_seqs:
            return None

        visited_state_bins = set()
        visited_behavior_bins = set()
        visited_fault_bins = set()

        history = {
            'episodes': [],
            'state_coverage': [],
            'behavior_diversity': [],
            'fault_diversity': []
        }

        total_eps = len(self.obs_seqs)
        print(f"Processing metrics for {total_eps} episodes...")

        for i in range(total_eps):
            sequence = self.obs_seqs[i]
            log_entry = self.logs[i]
            is_crash = log_entry.get('did_crash', False)

            # 1. 计算状态覆盖 (State Coverage)
            for state in sequence:
                # 容错处理：确保 state 有两个元素
                if len(state) >= 2:
                    pos, vel = state[0], state[1]
                    idx = self._get_grid_index(
                        (pos, vel), 
                        (self.state_pos_range, self.state_vel_range), 
                        self.state_grid_size
                    )
                    visited_state_bins.add(idx)

            # 2. 计算行为多样性 (Behavior Diversity)
            bd_values = self._calc_behavior_descriptor(sequence)
            bd_idx = self._get_grid_index(
                bd_values,
                (self.bd_max_pos_range, self.bd_avg_speed_range),
                self.behavior_grid_size
            )
            visited_behavior_bins.add(bd_idx)

            # 3. 计算故障多样性 (Fault Diversity)
            if is_crash:
                visited_fault_bins.add(bd_idx)

            history['episodes'].append(i + 1)
            history['state_coverage'].append(len(visited_state_bins))
            history['behavior_diversity'].append(len(visited_behavior_bins))
            history['fault_diversity'].append(len(visited_fault_bins))

        return history

def plot_metrics(history, save_path):
    if history is None: return

    episodes = history['episodes']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_config = [
        {
            'key': 'state_coverage',
            'title': 'State Coverage',
            'ylabel': '# Unique State Bins',
            'color': '#1f77b4', 
            'desc': 'Grid: Position × Velocity'
        },
        {
            'key': 'behavior_diversity',
            'title': 'Behavior Diversity',
            'ylabel': '# Unique Behavior Bins',
            'color': '#2ca02c', 
            'desc': 'Grid: MaxPos × AvgSpeed'
        },
        {
            'key': 'fault_diversity',
            'title': 'Fault Diversity',
            'ylabel': '# Unique Fault Bins',
            'color': '#d62728', 
            'desc': 'Grid: MaxPos × AvgSpeed (Crashes Only)'
        }
    ]

    for ax, config in zip(axes, metrics_config):
        data = history[config['key']]
        
        # Label 设为 G-Model
        ax.plot(episodes, data, color=config['color'], linewidth=2.5, label='G-Model')
        ax.fill_between(episodes, data, color=config['color'], alpha=0.1)
        
        ax.set_title(config['title'], fontweight='bold', pad=15)
        ax.set_xlabel('Episodes (Iterations)')
        ax.set_ylabel(config['ylabel'])
        
        ax.text(0.05, 0.95, config['desc'], transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right')

    plt.tight_layout()
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot G-Model Diversity Trends')
    parser.add_argument('--folder', type=str, default=None, 
                        help='Path to the results folder (e.g., results/MC_generative_...)')
    args = parser.parse_args()

    # 1. 确定数据文件夹
    if args.folder:
        target_dir = args.folder
    else:
        target_dir = get_latest_result_dir()
        if target_dir:
            print(f"Auto-detected latest result directory: {target_dir}")
        else:
            print("No g-model results directory found in ./results (looking for 'MC_*').")
            return

    # 2. 加载数据
    obs_seqs, logs = load_gmodel_data(target_dir)
    
    if not obs_seqs:
        return

    # 3. 初始化分析器 (50x50 网格)
    analyzer = DiversityAnalyzer(
        obs_seqs=obs_seqs, 
        logs=logs,
        state_grid_size=(50, 50),    
        behavior_grid_size=(50, 50)   
    )
    
    # 4. 计算指标
    history_data = analyzer.calculate_trends()
    
    # 5. 绘图
    plot_save_path = os.path.join(target_dir, PLOT_FILE)
    plot_metrics(history_data, save_path=plot_save_path)
    print(f"Done! Plot saved to {plot_save_path}")

if __name__ == "__main__":
    main()