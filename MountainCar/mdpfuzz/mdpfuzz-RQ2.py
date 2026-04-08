import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --- 配置区域 ---
OBS_FILE = 'logs/MC_DQN_NoCov_5_0.01_0.1_0_9000it_obs.txt' 
# 自动推断 Log 文件路径 (假设文件名仅后缀不同)
LOG_FILE = OBS_FILE.replace('_obs.txt', '_logs.txt')

LABEL = 'MDPFuzz'
IS_RT = False 

# 输出图片名称
PLOT_DIVERSITY_FILE = 'RQ2_Diversity_Filtered.png'
PLOT_SEEDS_FILE = 'RQ2_Crash_Seeds_Distribution.png'

# 绘图风格设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5
})

class MdpFuzzLogParser:
    """解析 _obs.txt 文件，用于多样性分析"""
    def __init__(self, filepath, is_rt=False):
        self.filepath = filepath
        self.is_rt = is_rt

    def parse(self):
        print(f"Parsing Obs file: {self.filepath} ...")
        print(f"Mode: {'Random Testing (Keep All)' if self.is_rt else 'MDPFuzz (Exclude Init Phase/Gen 0)'}")
        count = 0
        skipped = 0
        try:
            with open(self.filepath, 'r') as f:
                current_info = None
                current_data = []
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if line.startswith("--- Test Case Info:"):
                        if current_info is not None:
                            gen = current_info.get('Generation', 0)
                            if not self.is_rt and gen == 0:
                                skipped += 1
                            else:
                                yield current_info, np.array(current_data)
                                count += 1
                        
                        json_part = line[len("--- Test Case Info: "):-len(" ---")]
                        current_info = json.loads(json_part)
                        current_data = []
                        
                    else:
                        if current_info is not None:
                            vals = [float(x) for x in line.split(',')]
                            current_data.append(vals)
                            
                if current_info is not None:
                    gen = current_info.get('Generation', 0)
                    if not self.is_rt and gen == 0:
                        skipped += 1
                    else:
                        yield current_info, np.array(current_data)
                        count += 1
        except FileNotFoundError:
            print(f"Error: Obs file not found at {self.filepath}")
                    
        print(f"  - Valid Episodes Used: {count}")
        print(f"  - Skipped (Init Phase): {skipped}")

class SeedCrashAnalyzer:
    """新增：解析 _logs.txt 文件，用于统计初始种子导致的 Crash"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.delimiter = '; ' # 与 logger.py 中定义的一致

    def analyze(self):
        print(f"\nParsing Log file for Seeds: {self.filepath} ...")
        seed_crashes = []
        try:
            with open(self.filepath, 'r') as f:
                header_line = f.readline().strip()
                if not header_line:
                    return {}
                headers = header_line.split(self.delimiter)
                
                # 检查是否存在 SeedID 列
                if 'SeedID' not in headers:
                    print("Warning: 'SeedID' column not found in logs. Please ensure you are using the modified logger.")
                    return {}
                if 'Oracle' not in headers:
                    print("Warning: 'Oracle' column not found.")
                    return {}

                idx_oracle = headers.index('Oracle')
                idx_seed = headers.index('SeedID')

                for line in f:
                    values = [v.strip() for v in line.strip().split(self.delimiter)]
                    # 确保行数据完整
                    if len(values) <= max(idx_oracle, idx_seed):
                        continue

                    oracle_str = values[idx_oracle]
                    seed_id_str = values[idx_seed]

                    # 只统计发生了 Crash 且拥有有效 SeedID 的条目
                    if oracle_str == 'True' and seed_id_str != 'None':
                        try:
                            seed_crashes.append(int(seed_id_str))
                        except ValueError:
                            pass
                            
        except FileNotFoundError:
            print(f"Error: Log file not found at {self.filepath}")
            return {}

        counts = Counter(seed_crashes)
        print(f"  - Total Crashes Linked to Seeds: {len(seed_crashes)}")
        print(f"  - Unique Initial Seeds Causing Crashes: {len(counts)}")
        return counts

class DiversityAnalyzer:
    def __init__(self, state_grid_size=(50, 50), behavior_grid_size=(50, 50)):
        self.state_grid_size = state_grid_size
        self.behavior_grid_size = behavior_grid_size
        
        self.state_pos_range = (-1.2, 0.6)
        self.state_vel_range = (-0.07, 0.07)
        
        self.bd_max_pos_range = (-1.2, 0.6)
        self.bd_avg_speed_range = (0.0, 0.05)

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
        if len(seq_arr) == 0: return -1.2, 0.0
        positions = seq_arr[:, 0]
        velocities = seq_arr[:, 1]
        
        max_pos = np.max(positions)
        avg_speed = np.mean(np.abs(velocities))
        return max_pos, avg_speed

    def calculate_trends(self, parser):
        visited_state_bins = set()
        visited_behavior_bins = set()
        visited_fault_bins = set()

        history = {
            'episodes': [],
            'state_coverage': [],
            'behavior_diversity': [],
            'fault_diversity': []
        }

        episode_idx = 0
        for info, obs_seq in parser.parse():
            episode_idx += 1
            is_crash = info.get('Oracle', False)
            for state in obs_seq:
                if len(state) >= 2:
                    pos, vel = state[0], state[1]
                    idx = self._get_grid_index(
                        (pos, vel), 
                        (self.state_pos_range, self.state_vel_range), 
                        self.state_grid_size
                    )
                    visited_state_bins.add(idx)
            bd_values = self._calc_behavior_descriptor(obs_seq)
            bd_idx = self._get_grid_index(
                bd_values,
                (self.bd_max_pos_range, self.bd_avg_speed_range),
                self.behavior_grid_size
            )
            visited_behavior_bins.add(bd_idx)
            if is_crash:
                visited_fault_bins.add(bd_idx)
            history['episodes'].append(episode_idx)
            history['state_coverage'].append(len(visited_state_bins))
            history['behavior_diversity'].append(len(visited_behavior_bins))
            history['fault_diversity'].append(len(visited_fault_bins))
        return history

def plot_metrics(history, save_path, label, is_rt):
    if not history or not history['episodes']:
        print("No diversity data to plot.")
        return

    episodes = history['episodes']
    plot_color = '#1f77b4' if is_rt else '#ff7f0e'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_config = [
        {
            'key': 'state_coverage',
            'title': 'State Coverage',
            'ylabel': '# Unique State Bins',
            'desc': 'Grid: Pos × Vel'
        },
        {
            'key': 'behavior_diversity',
            'title': 'Behavior Diversity',
            'ylabel': '# Unique Behavior Bins',
            'desc': 'Grid: MaxPos × AvgSpeed'
        },
        {
            'key': 'fault_diversity',
            'title': 'Fault Diversity',
            'ylabel': '# Unique Fault Bins',
            'desc': 'Crashes Only'
        }
    ]

    for ax, config in zip(axes, metrics_config):
        data = history[config['key']]
        ax.plot(episodes, data, color=plot_color, label=label)
        ax.fill_between(episodes, data, color=plot_color, alpha=0.1)
        ax.set_title(config['title'], fontweight='bold', pad=15)
        ax.set_xlabel('Fuzzing Episodes (Post-Init)')
        ax.set_ylabel(config['ylabel'])
        ax.text(0.05, 0.95, config['desc'], transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(loc='lower right')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    print(f"Saving diversity plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_seed_crash_distribution(seed_counts, save_path):
    """新增：绘制初始种子导致Crash的分布图"""
    if not seed_counts:
        print("No seed crash data to plot.")
        return

    # 按 SeedID 排序，以便X轴有序
    sorted_seeds = sorted(seed_counts.keys())
    counts = [seed_counts[s] for s in sorted_seeds]
    
    # 如果种子数量太多，只显示 Top N 或者调整图表大小
    plt.figure(figsize=(12, 6))
    
    # 绘制柱状图
    bars = plt.bar(sorted_seeds, counts, color='#d62728', alpha=0.8, edgecolor='black')
    
    plt.xlabel('Initial Seed ID')
    plt.ylabel('Number of Crashes Derived')
    plt.title(f'Distribution of Crashes by Initial Seed ({LABEL})', fontweight='bold')
    
    # 只有当种子数量不多时，才强制显示所有 X 轴刻度
    if len(sorted_seeds) <= 30:
        plt.xticks(sorted_seeds)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 在柱子上显示具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}', ha='center', va='bottom', fontsize=9)
                 
    plt.tight_layout()
    print(f"Saving seed distribution plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.show()

def main():
    print(f"Target Obs File: {OBS_FILE}")
    print(f"Target Log File: {LOG_FILE}")
    
    # 1. 多样性分析 (原有功能)
    parser = MdpFuzzLogParser(OBS_FILE, is_rt=IS_RT)
    analyzer = DiversityAnalyzer(
        state_grid_size=(50, 50),
        behavior_grid_size=(50, 50)
    )
    history = analyzer.calculate_trends(parser)
    plot_metrics(history, save_path=PLOT_DIVERSITY_FILE, label=LABEL, is_rt=IS_RT)

    # 2. 初始种子Crash分布分析 (新增功能)
    if not IS_RT: # Random Testing 没有 SeedID 血缘概念，通常不绘制此图
        seed_analyzer = SeedCrashAnalyzer(LOG_FILE)
        seed_counts = seed_analyzer.analyze()
        plot_seed_crash_distribution(seed_counts, save_path=PLOT_SEEDS_FILE)
    else:
        print("\nSkipping Seed Analysis for Random Testing mode (SeedID not applicable).")

if __name__ == "__main__":
    main()