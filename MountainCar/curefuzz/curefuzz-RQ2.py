import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# --- 1. 增加 Crash 文件路径 ---
OBS_FILE = 'obs_sequences.pkl'
LOG_FILE = 'selection_log.pkl'
CRASH_FILE = 'cure_crash.pkl'  # 新增：读取 crash 结果文件
PLOT_FILE = 'diversity_metrics_plot.png'

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
    def __init__(self, obs_path, log_path, crash_path,  # 修改：传入 crash_path
                 state_grid_size=(100, 100), 
                 behavior_grid_size=(100, 100)):
        
        self.obs_path = obs_path
        self.log_path = log_path
        self.crash_path = crash_path
        
        self.state_grid_size = state_grid_size
        self.behavior_grid_size = behavior_grid_size
        
        self.state_pos_range = (-1.2, 0.6)
        self.state_vel_range = (-0.07, 0.07)
        
        self.bd_max_pos_range = (-1.2, 0.6)
        self.bd_avg_speed_range = (0.0, 0.05)
        
        self.obs_seqs = self._load_pickle(obs_path)
        self.logs = self._load_pickle(log_path)
        self.crashes = self._load_pickle(crash_path) # 加载 crash 数据

    def _load_pickle(self, filepath):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return []
        with open(filepath, 'rb') as f:
            print(f"Loaded data from {filepath}")
            return pickle.load(f)

    def _get_grid_index(self, values, ranges, grid_size):
        indices = []
        for val, (min_val, max_val), bins in zip(values, ranges, grid_size):
            norm = (val - min_val) / (max_val - min_val)
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

    # --- 2. 新增：统计 Crash 来源的方法 ---
    def get_crash_statistics(self):
        if not self.crashes:
            return {'total': 0, 'unique_seeds': 0}
        
        total_crashes = len(self.crashes)
        unique_seeds = set()
        
        for c in self.crashes:
            # 兼容处理：如果包含 root_id 则使用，否则忽略
            if isinstance(c, dict) and 'root_id' in c:
                if c['root_id'] is not None:
                    unique_seeds.add(c['root_id'])
            else:
                # 如果是旧数据或者没 ID，无法统计唯一性
                pass
                
        return {
            'total': total_crashes,
            'unique_seeds': len(unique_seeds)
        }

    def calculate_trends(self):
        if not self.obs_seqs or not self.logs:
            print("No data loaded for trends! Please check obs/log paths.")
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

        min_len = min(len(self.obs_seqs), len(self.logs))
        print(f"Processing {min_len} episodes...")

        for i in range(min_len):
            sequence = self.obs_seqs[i]
            log_entry = self.logs[i]
            is_crash = log_entry.get('did_crash', False)

            for state in sequence:
                pos, vel = state
                idx = self._get_grid_index(
                    (pos, vel), 
                    (self.state_pos_range, self.state_vel_range), 
                    self.state_grid_size
                )
                visited_state_bins.add(idx)

            bd_values = self._calc_behavior_descriptor(sequence)
            bd_idx = self._get_grid_index(
                bd_values,
                (self.bd_max_pos_range, self.bd_avg_speed_range),
                self.behavior_grid_size
            )
            visited_behavior_bins.add(bd_idx)

            if is_crash:
                visited_fault_bins.add(bd_idx)

            history['episodes'].append(i + 1)
            history['state_coverage'].append(len(visited_state_bins))
            history['behavior_diversity'].append(len(visited_behavior_bins))
            history['fault_diversity'].append(len(visited_fault_bins))

        return history

def plot_metrics(history, crash_stats, save_path): # 增加 crash_stats 参数
    if history is None: return

    episodes = history['episodes']
    
    # --- 3. 修改布局为 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten() # 展平为 [0, 1, 2, 3] 以便索引
    
    # 定义前三个折线图的配置
    metrics_config = [
        {
            'key': 'state_coverage',
            'title': 'State Coverage',
            'ylabel': '# Unique State Bins',
            'color': '#1f77b4', 
            'desc': 'Grid: Position × Velocity',
            'ax_idx': 0
        },
        {
            'key': 'behavior_diversity',
            'title': 'Behavior Diversity',
            'ylabel': '# Unique Behavior Bins',
            'color': '#2ca02c', 
            'desc': 'Grid: MaxPos × AvgSpeed',
            'ax_idx': 1
        },
        {
            'key': 'fault_diversity',
            'title': 'Fault Diversity',
            'ylabel': '# Unique Fault Bins',
            'color': '#d62728', 
            'desc': 'Grid: MaxPos × AvgSpeed (Crashes Only)',
            'ax_idx': 2
        }
    ]

    # 绘制前三个折线图
    for config in metrics_config:
        ax = axes[config['ax_idx']]
        data = history[config['key']]
        
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

    # --- 4. 绘制第四个图：Crash 来源统计柱状图 ---
    ax_bar = axes[3]
    labels = ['Total Crashes', 'Unique Source Seeds']
    values = [crash_stats['total'], crash_stats['unique_seeds']]
    colors = ['#d62728', '#9467bd'] # 红色代表总数，紫色代表独立种子数
    
    bars = ax_bar.bar(labels, values, color=colors, alpha=0.8, width=0.5)
    
    ax_bar.set_title('Crash Source Analysis', fontweight='bold', pad=15)
    ax_bar.set_ylabel('Count')
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 在柱子上显示具体数值
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    # 添加说明文字
    desc_text = (f"Total Crashes: {values[0]}\n"
                 f"Unique Seeds: {values[1]}\n"
                 f"Avg Crashes/Seed: {values[0]/values[1] if values[1]>0 else 0:.1f}")
    
    ax_bar.text(0.95, 0.95, desc_text, transform=ax_bar.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.show()

def main():
    print(f"Target Obs File: {OBS_FILE}")
    print(f"Target Log File: {LOG_FILE}")
    print(f"Target Crash File: {CRASH_FILE}")

    analyzer = DiversityAnalyzer(
        obs_path=OBS_FILE, 
        log_path=LOG_FILE,
        crash_path=CRASH_FILE,  # 传入文件路径
        state_grid_size=(50, 50),    
        behavior_grid_size=(50, 50)   
    )
    
    history_data = analyzer.calculate_trends()
    crash_stats = analyzer.get_crash_statistics() # 获取统计数据
    
    print(f"Crash Statistics: {crash_stats}")
    
    plot_metrics(history_data, crash_stats, save_path=PLOT_FILE)

if __name__ == "__main__":
    main()