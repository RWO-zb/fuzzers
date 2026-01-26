import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

# --- 配置 ---
OBS_FILE = 'all_episodes_obs.txt'
LOG_FILE = 'all_run_seeds_0.pkl'
PLOT_FILE = 'SeqFuzz_RQ2_Diversity_With_Seeds.png'

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
        print(f"Processing {min_len} episodes for trends...")
        
        valid_ep_count = 0

        for i in range(min_len):
            sequence = self.obs_seqs[i]
            log_entry = self.logs[i]
            
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

def plot_metrics(history, logs, save_path):
    if not history['episodes']: 
        print("No valid data to plot.")
        return

    episodes = history['episodes']
    
    # --- 修改布局为 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten() # 展平为 [0, 1, 2, 3] 以便索引
    
    # 1. State Coverage (Top-Left)
    ax = axes[0]
    data = history['state_coverage']
    ax.plot(episodes, data, color='#1f77b4', linewidth=2.5)
    ax.fill_between(episodes, data, color='#1f77b4', alpha=0.1)
    ax.set_title('State Coverage', fontweight='bold', pad=10)
    ax.set_xlabel('Fuzz Episodes')
    ax.set_ylabel('# Unique State Bins')
    ax.grid(True, linestyle='--', alpha=0.7)

    # 2. Behavior Diversity (Top-Right)
    ax = axes[1]
    data = history['behavior_diversity']
    ax.plot(episodes, data, color='#2ca02c', linewidth=2.5)
    ax.fill_between(episodes, data, color='#2ca02c', alpha=0.1)
    ax.set_title('Behavior Diversity', fontweight='bold', pad=10)
    ax.set_xlabel('Fuzz Episodes')
    ax.set_ylabel('# Unique Behavior Bins')
    ax.grid(True, linestyle='--', alpha=0.7)

    # 3. Fault Diversity (Bottom-Left)
    ax = axes[2]
    data = history['fault_diversity']
    ax.plot(episodes, data, color='#d62728', linewidth=2.5)
    ax.fill_between(episodes, data, color='#d62728', alpha=0.1)
    ax.set_title('Fault Diversity', fontweight='bold', pad=10)
    ax.set_xlabel('Fuzz Episodes')
    ax.set_ylabel('# Unique Fault Bins')
    ax.grid(True, linestyle='--', alpha=0.7)

    # 4. [新增] Crash Root Seed Distribution (Bottom-Right)
    ax = axes[3]
    
    # 提取并处理种子数据
    crash_seeds = []
    for entry in logs:
        if entry.get('crashed', False):
            seed = entry.get('root_seed', None)
            if seed is not None:
                # 转换 numpy array/list 为 tuple 以便作为字典键
                if isinstance(seed, (np.ndarray, list)):
                    # 保留4位小数防止浮点误差
                    seed_tuple = tuple(np.round(np.array(seed), 4))
                    crash_seeds.append(seed_tuple)
                else:
                    crash_seeds.append(seed)
    
    if crash_seeds:
        # 统计每个种子导致的 Crash 次数
        seed_counts = Counter(crash_seeds)
        # 按次数从高到低排序
        sorted_seeds = sorted(seed_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 准备绘图数据
        labels = [f"Seed {i+1}" for i in range(len(sorted_seeds))]
        counts = [count for _, count in sorted_seeds]
        
        # 绘制柱状图
        bars = ax.bar(labels, counts, color='#9467bd', alpha=0.8, edgecolor='black')
        
        # 标题和标签
        ax.set_title(f'Crashes per Root Seed (Total Unique: {len(sorted_seeds)})', fontweight='bold', pad=10)
        ax.set_ylabel('# Crashes Generated')
        ax.set_xlabel('Root Seeds (Sorted by Impact)')
        
        # 优化 X 轴标签显示
        if len(labels) > 15:
            # 如果柱子太多，隐藏具体的 x 轴标签，防止重叠
            ax.set_xticks([])
        else:
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
        # 在柱子上方显示数值（如果柱子不太密）
        if len(labels) < 20:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No Crashes Found', 
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Root Seed Distribution')

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

def main():
    obs, logs = load_data()
    if logs:
        analyzer = DiversityAnalyzer(obs, logs, state_grid_size=(50, 50), behavior_grid_size=(50, 50))
        history = analyzer.calculate_trends()
        # 将 logs 传递给 plot_metrics 以便绘制柱状图
        plot_metrics(history, logs, PLOT_FILE)

if __name__ == "__main__":
    main()