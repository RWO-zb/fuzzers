import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd  # 新增: 用于读取CSV统计种子数据

OBS_FILE = 'seed42/mc_test_obs.txt'
CSV_FILE = 'seed42/mc_test_data.csv'  # 新增: CSV文件路径
PLOT_NAME = 'QDFuzz_RQ2.png'
STATE_GRID_SIZE = (50, 50)
BEHAVIOR_GRID_SIZE = (50, 50)

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

def load_rq2_data_fuzz_only(obs_path):
    print(f"Loading data from {obs_path} (Fuzzing Phase Only)...")
    if not os.path.exists(obs_path):
        print("Error: OBS file not found.")
        return [], []
    logs = []
    obs_seqs = []
    current_seq = []
    current_info = None
    with open(obs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('--- Test Case Info:'):
                if current_info is not None and current_seq:
                    gen = current_info.get('Generation', 0)
                    if gen > 0:
                        obs_seqs.append(np.array(current_seq))
                        logs.append({'did_crash': bool(current_info.get('Oracle', False))})
                current_seq = []
                try:
                    json_str = line.split('--- Test Case Info:')[1].rsplit('---', 1)[0].strip()
                    current_info = json.loads(json_str)
                except:
                    current_info = None
                    
            else:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        current_seq.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue
        if current_info is not None and current_seq:
            gen = current_info.get('Generation', 0)
            if gen > 0:
                obs_seqs.append(np.array(current_seq))
                logs.append({'did_crash': bool(current_info.get('Oracle', False))})
    print(f"Loaded {len(logs)} valid episodes from Fuzzing Phase.")
    return logs, obs_seqs

# 新增函数: 读取CSV并统计导致Crash的种子
def load_seed_crash_stats(csv_path):
    print(f"Loading seed stats from {csv_path}...")
    if not os.path.exists(csv_path):
        print("Warning: CSV file not found. Cannot plot seed stats.")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        # 检查是否存在 seed_id 列
        if 'seed_id' not in df.columns:
            print("Warning: 'seed_id' column not found in CSV. Did you run the modified framework?")
            return {}
            
        # 筛选出 Crash 的样本
        crashes = df[df['is_faulty'] == True]
        if crashes.empty:
            print("No crashes found in CSV.")
            return {}
            
        # 统计每个种子导致的 Crash 数量
        seed_counts = crashes['seed_id'].value_counts().to_dict()
        print(f"Found {len(seed_counts)} unique seeds that caused crashes.")
        return seed_counts
    except Exception as e:
        print(f"Error loading CSV for seed stats: {e}")
        return {}

class DiversityAnalyzer:
    def __init__(self, obs_seqs, logs):
        self.obs_seqs = obs_seqs
        self.logs = logs
        self.state_grid_size = STATE_GRID_SIZE
        self.behavior_grid_size = BEHAVIOR_GRID_SIZE
        self.state_pos_range = (-1.2, 0.6)
        self.state_vel_range = (-0.07, 0.07)
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
        if len(seq_arr) == 0: return -1.2, 0.0
        return np.max(seq_arr[:, 0]), np.mean(np.abs(seq_arr[:, 1]))

    def calculate_trends(self):
        visited_state_bins = set()
        visited_behavior_bins = set()
        visited_fault_bins = set()
        history = {'episodes': [], 'state_coverage': [], 'behavior_diversity': [], 'fault_diversity': []}
        for i in range(len(self.obs_seqs)):
            sequence = self.obs_seqs[i]
            log_entry = self.logs[i]
            for state in sequence:
                idx = self._get_grid_index(state, (self.state_pos_range, self.state_vel_range), self.state_grid_size)
                visited_state_bins.add(idx)
            bd_values = self._calc_behavior_descriptor(sequence)
            bd_idx = self._get_grid_index(bd_values, (self.bd_max_pos_range, self.bd_avg_speed_range), self.behavior_grid_size)
            visited_behavior_bins.add(bd_idx)
            if log_entry.get('did_crash', False):
                visited_fault_bins.add(bd_idx)
            history['episodes'].append(i + 1)
            history['state_coverage'].append(len(visited_state_bins))
            history['behavior_diversity'].append(len(visited_behavior_bins))
            history['fault_diversity'].append(len(visited_fault_bins))
        return history

def plot_rq2(history, seed_counts):
    episodes = history['episodes']
    # [修改] 布局改为 2x2 以容纳新的柱状图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # 展平 axes 方便索引: 0,1,2 是折线图, 3 是新的柱状图
    ax_list = axes.flatten()
    
    metrics_config = [
        {'key': 'state_coverage', 'title': 'State Coverage', 'ylabel': '# Unique State Bins', 'color': '#1f77b4', 'desc': 'Grid: Pos × Vel'},
        {'key': 'behavior_diversity', 'title': 'Behavior Diversity', 'ylabel': '# Unique Behavior Bins', 'color': '#2ca02c', 'desc': 'Grid: MaxPos × AvgSpeed'},
        {'key': 'fault_diversity', 'title': 'Fault Diversity', 'ylabel': '# Unique Fault Bins', 'color': '#d62728', 'desc': 'Grid: MaxPos × AvgSpeed (Crashes)'}
    ]
    
    # 绘制前三个折线图
    for i, config in enumerate(metrics_config):
        ax = ax_list[i]
        data = history[config['key']]
        ax.plot(episodes, data, color=config['color'], linewidth=2.5, label='QDFuzz (Fuzz Only)')
        ax.fill_between(episodes, data, color=config['color'], alpha=0.1)
        ax.set_title(config['title'], fontweight='bold', pad=15)
        ax.set_xlabel('Fuzzing Episodes')
        ax.set_ylabel(config['ylabel'])
        ax.text(0.05, 0.95, config['desc'], transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.7)

    # [新增] 绘制第四个图：导致Crash的初始种子统计 (柱状图)
    ax_bar = ax_list[3]
    if seed_counts:
        # 排序：按Crash数量降序
        sorted_seeds = sorted(seed_counts.items(), key=lambda x: x[1], reverse=True)
        seeds = [str(k) for k, v in sorted_seeds] # Seed ID 转字符串
        counts = [v for k, v in sorted_seeds]
        
        # 只展示前20个，如果太多的话
        if len(seeds) > 20:
            seeds = seeds[:20]
            counts = counts[:20]
            title_suffix = " (Top 20)"
        else:
            title_suffix = ""
            
        bars = ax_bar.bar(seeds, counts, color='#9467bd', alpha=0.8, edgecolor='black')
        ax_bar.set_title(f"Crashes per Initial Seed{title_suffix}", fontweight='bold', pad=15)
        ax_bar.set_xlabel('Seed ID')
        ax_bar.set_ylabel('Number of Derived Crashes')
        ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
            
        # 显示总数统计
        total_unique = len(seed_counts)
        ax_bar.text(0.95, 0.95, f'Total Unique Seeds\nCausing Crashes: {total_unique}', 
                   transform=ax_bar.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    else:
        ax_bar.text(0.5, 0.5, 'No Crash/Seed Data Available', ha='center', va='center')
        ax_bar.set_title("Crashes per Initial Seed")

    plt.tight_layout()
    plt.savefig(PLOT_NAME, dpi=300)
    print(f"Plot saved to {PLOT_NAME}")
    plt.show()

if __name__ == "__main__":
    logs, obs_seqs = load_rq2_data_fuzz_only(OBS_FILE)
    seed_counts = load_seed_crash_stats(CSV_FILE) # 加载CSV统计数据
    
    if logs and obs_seqs:
        analyzer = DiversityAnalyzer(obs_seqs, logs)
        history = analyzer.calculate_trends()
        plot_rq2(history, seed_counts) # 传入 seed_counts
    else:
        print("Skipping RQ2 plot due to missing data.")