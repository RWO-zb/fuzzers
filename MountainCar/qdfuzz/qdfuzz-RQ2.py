import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
import os

# --- 配置路径 ---
# 对应 RL_MountainCar/run_experiment.py 生成的文件路径
CSV_FILE = 'results/mc_test_data.csv'
OBS_FILE = 'results/mc_test_obs.txt'
PLOT_FILE = 'diversity_metrics_plot.png'

# --- 绘图样式设置 (参考 curefuzz-RQ2.py) ---
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

# --- 数据加载模块 (改编自 RL_MountainCar/qdfuzzplot.py) ---
def load_data(csv_path, obs_path):
    print(f"Loading data from {csv_path} and {obs_path}...")
    
    # 虽然 curefuzz-RQ2 主要需要轨迹和崩溃信息，
    # 但我们这里复用加载逻辑以确保兼容性。
    # 这里主要依赖 OBS_FILE 中的轨迹数据。
    
    obs_data = []
    current_seq = []
    current_is_crash = False 
    has_header = False
    
    if not os.path.exists(obs_path):
        print(f"Error: Observation file not found at {obs_path}")
        return []

    with open(obs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith('--- Test Case Info:'):
                # 如果已经读取过一段序列，则保存它
                if has_header and current_seq: 
                    obs_data.append({
                        'trajectory': np.array(current_seq),
                        'is_crash': current_is_crash
                    })
                
                # 重置状态
                current_seq = []
                has_header = True
                
                # 解析 JSON 获取 Oracle (是否崩溃)
                try:
                    json_str = line.split('--- Test Case Info:')[1].rsplit('---', 1)[0].strip()
                    info = json.loads(json_str)
                    current_is_crash = bool(info.get('Oracle', False))
                except Exception as e:
                    print(f"Warning: Failed to parse header: {e}")
                    current_is_crash = False
                    
            else:
                # 解析坐标 (Position, Velocity)
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        current_seq.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue
        
        # 保存最后一段序列
        if has_header and current_seq:
            obs_data.append({
                'trajectory': np.array(current_seq),
                'is_crash': current_is_crash
            })

    print(f"Loaded {len(obs_data)} episodes.")
    return obs_data

# --- 分析模块 (改编自 curefuzz-RQ2.py) ---
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
        
        # 行为描述符范围 (参考 curefuzz-RQ2)
        self.bd_max_pos_range = (-1.2, 0.6)
        self.bd_avg_speed_range = (0.0, 0.05) # 可以根据实际数据调整上限，例如 0.07

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
        positions = seq_arr[:, 0]
        velocities = seq_arr[:, 1]
        
        max_pos = np.max(positions)
        avg_speed = np.mean(np.abs(velocities))
        return max_pos, avg_speed

    def calculate_trends(self):
        if not self.obs_seqs or not self.logs:
            print("No data to process!")
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

        # 确保数据长度一致
        min_len = min(len(self.obs_seqs), len(self.logs))
        print(f"Processing metrics for {min_len} episodes...")

        for i in range(min_len):
            sequence = self.obs_seqs[i]
            log_entry = self.logs[i]
            is_crash = log_entry.get('did_crash', False)

            # 1. 计算状态覆盖 (State Coverage) - 遍历轨迹中的每个点
            for state in sequence:
                pos, vel = state
                idx = self._get_grid_index(
                    (pos, vel), 
                    (self.state_pos_range, self.state_vel_range), 
                    self.state_grid_size
                )
                visited_state_bins.add(idx)

            # 2. 计算行为多样性 (Behavior Diversity) - 每集一个特征
            bd_values = self._calc_behavior_descriptor(sequence)
            bd_idx = self._get_grid_index(
                bd_values,
                (self.bd_max_pos_range, self.bd_avg_speed_range),
                self.behavior_grid_size
            )
            visited_behavior_bins.add(bd_idx)

            # 3. 计算故障多样性 (Fault Diversity) - 仅统计崩溃的剧集
            if is_crash:
                visited_fault_bins.add(bd_idx)

            history['episodes'].append(i + 1)
            history['state_coverage'].append(len(visited_state_bins))
            history['behavior_diversity'].append(len(visited_behavior_bins))
            history['fault_diversity'].append(len(visited_fault_bins))

        return history

# --- 绘图模块 (完全复用 curefuzz-RQ2.py) ---
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
        
        # 这里的 label 可以根据需要修改，例如 'MountainCar-DQN'
        ax.plot(episodes, data, color=config['color'], linewidth=2.5, label='CureFuzz')
        
        ax.fill_between(episodes, data, color=config['color'], alpha=0.1)
        
        ax.set_title(config['title'], fontweight='bold', pad=15)
        ax.set_xlabel('Episodes (Iterations)')
        ax.set_ylabel(config['ylabel'])
        
        ax.text(0.05, 0.95, config['desc'], transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.7)
        # 如果需要图例，可以取消注释下面这行
        # ax.legend(loc='lower right')

    plt.tight_layout()
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.show()

# --- 主函数 ---
def main():
    # 1. 加载数据
    obs_data = load_data(CSV_FILE, OBS_FILE)
    
    if not obs_data:
        print("Failed to load data. Please check if 'results/mc_test_obs.txt' exists.")
        return

    # 2. 转换数据格式以适配 Analyzer
    # Analyzer 需要 obs_seqs (轨迹列表) 和 logs (包含 did_crash 的字典列表)
    obs_seqs = [d['trajectory'] for d in obs_data]
    logs = [{'did_crash': d['is_crash']} for d in obs_data]
    
    # 3. 初始化分析器 (使用与 curefuzz-RQ2 相同的网格参数 50x50)
    analyzer = DiversityAnalyzer(
        obs_seqs=obs_seqs, 
        logs=logs,
        state_grid_size=(50, 50),    
        behavior_grid_size=(50, 50)   
    )
    
    # 4. 计算指标趋势
    history_data = analyzer.calculate_trends()
    
    # 5. 绘图
    plot_metrics(history_data, save_path=PLOT_FILE)

if __name__ == "__main__":
    main()