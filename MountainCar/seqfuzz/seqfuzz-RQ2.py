import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置 ---
OBS_FILE = 'all_episodes_obs.txt'
LOG_FILE = 'all_run_seeds_0.pkl'
PLOT_FILE = 'SeqFuzz_RQ2_Diversity.png'

# --- 绘图样式 ---
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

class DiversityAnalyzer:
    def __init__(self, obs_seqs, logs, 
                 state_grid_size=(100, 100), 
                 behavior_grid_size=(100, 100)):
        
        self.obs_seqs = obs_seqs
        self.logs = logs
        self.state_grid_size = state_grid_size
        self.behavior_grid_size = behavior_grid_size
        
        # MountainCar 标准范围
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
        if len(seq_arr) == 0:
            return -1.2, 0.0
        positions = seq_arr[:, 0]
        velocities = seq_arr[:, 1]
        max_pos = np.max(positions)
        avg_speed = np.mean(np.abs(velocities))
        return max_pos, avg_speed

    def calculate_trends(self):
        visited_state_bins = set()
        visited_behavior_bins = set()
        visited_fault_bins = set()

        history = {
            'episodes': [], 'state_coverage': [], 
            'behavior_diversity': [], 'fault_diversity': []
        }

        # 确保 obs 和 logs 长度对齐
        min_len = min(len(self.obs_seqs), len(self.logs))
        print(f"Processing {min_len} episodes...")
        
        valid_ep_count = 0

        for i in range(min_len):
            sequence = self.obs_seqs[i]
            log_entry = self.logs[i]
            
            # enjoy.py 只记录 Fuzz 阶段数据，因此这里的每条数据都是 Fuzz 数据
            # 无需像之前那样检查 parent_depth
            
            valid_ep_count += 1
            # 字段修正: did_crash -> crashed
            is_crash = log_entry.get('crashed', False)

            # 1. State Coverage
            for state in sequence:
                pos, vel = state
                idx = self._get_grid_index(
                    (pos, vel), 
                    (self.state_pos_range, self.state_vel_range), 
                    self.state_grid_size
                )
                visited_state_bins.add(idx)

            # 2. Behavior Diversity
            bd_values = self._calc_behavior_descriptor(sequence)
            bd_idx = self._get_grid_index(
                bd_values,
                (self.bd_max_pos_range, self.bd_avg_speed_range),
                self.behavior_grid_size
            )
            visited_behavior_bins.add(bd_idx)

            # 3. Fault Diversity
            if is_crash:
                visited_fault_bins.add(bd_idx)

            history['episodes'].append(valid_ep_count)
            history['state_coverage'].append(len(visited_state_bins))
            history['behavior_diversity'].append(len(visited_behavior_bins))
            history['fault_diversity'].append(len(visited_fault_bins))

        return history

def load_data():
    # 加载 Pickle 日志
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        return [], []
    with open(LOG_FILE, 'rb') as f:
        logs = pickle.load(f)
    print(f"Loaded {len(logs)} logs.")

    # 加载 TXT 观测序列
    if not os.path.exists(OBS_FILE):
        print(f"Obs file not found: {OBS_FILE}")
        return [], []
    
    obs_seqs = []
    current_seq = []
    print(f"Parsing {OBS_FILE}...")
    with open(OBS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if '######' in line:
                if current_seq:
                    obs_seqs.append(np.array(current_seq))
                    current_seq = []
            else:
                try:
                    parts = line.strip(',').split(',')
                    vals = [float(p) for p in parts if p.strip()]
                    if len(vals) >= 2: current_seq.append(vals[:2])
                except: continue
    if current_seq: obs_seqs.append(np.array(current_seq))
    print(f"Loaded {len(obs_seqs)} sequences.")
    
    return obs_seqs, logs

def plot_metrics(history, save_path):
    if not history['episodes']: 
        print("No valid data to plot.")
        return

    episodes = history['episodes']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_config = [
        {'key': 'state_coverage', 'title': 'State Coverage', 'ylabel': '# Unique State Bins', 'color': '#1f77b4', 'desc': 'Grid: Position × Velocity'},
        {'key': 'behavior_diversity', 'title': 'Behavior Diversity', 'ylabel': '# Unique Behavior Bins', 'color': '#2ca02c', 'desc': 'Grid: MaxPos × AvgSpeed'},
        {'key': 'fault_diversity', 'title': 'Fault Diversity', 'ylabel': '# Unique Fault Bins', 'color': '#d62728', 'desc': 'Grid: MaxPos × AvgSpeed (Crashes)'}
    ]

    for ax, config in zip(axes, metrics_config):
        data = history[config['key']]
        ax.plot(episodes, data, color=config['color'], linewidth=2.5, label='SeqFuzz')
        ax.fill_between(episodes, data, color=config['color'], alpha=0.1)
        ax.set_title(config['title'], fontweight='bold', pad=15)
        ax.set_xlabel('Fuzz Episodes')
        ax.set_ylabel(config['ylabel'])
        ax.text(0.05, 0.95, config['desc'], transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

def main():
    obs, logs = load_data()
    if logs:
        analyzer = DiversityAnalyzer(obs, logs, state_grid_size=(50, 50), behavior_grid_size=(50, 50))
        history = analyzer.calculate_trends()
        plot_metrics(history, PLOT_FILE)

if __name__ == "__main__":
    main()