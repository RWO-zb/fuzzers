import os
import json
import numpy as np
import matplotlib.pyplot as plt

OBS_FILE = 'MC_DQN_RT_0_5000it_obs.txt'   
PLOT_FILE = 'mdpfuzz_diversity_metrics_plot.png'

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

class MdpFuzzLogParser:
    def __init__(self, filepath):
        self.filepath = filepath

    def parse(self):
        print(f"Parsing file: {self.filepath} ...")
        count = 0
        if not os.path.exists(self.filepath):
            print(f"Error: File not found at {self.filepath}")
            return

        with open(self.filepath, 'r') as f:
            current_info = None
            current_data = []
            
            for line in f:
                line = line.strip()
                if not line: continue
                
                if line.startswith("--- Test Case Info:"):
                    if current_info is not None:
                        yield current_info, np.array(current_data)
                        count += 1
                        if count % 1000 == 0:
                            print(f"  Processed {count} episodes...", end='\r')
                    
                    json_part = line[len("--- Test Case Info: "):-len(" ---")]
                    current_info = json.loads(json_part)
                    current_data = []
                    
                else:
                    if current_info is not None:          
                        vals = [float(x) for x in line.split(',')]
                        current_data.append(vals)
                        
            if current_info is not None:
                yield current_info, np.array(current_data)
        print(f"\nFinished parsing. Total episodes: {count + 1}")

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

def plot_metrics(history, save_path):
    if not history or not history['episodes']:
        print("No data to plot.")
        return

    episodes = history['episodes']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_config = [
        {
            'key': 'state_coverage',
            'title': 'State Coverage',
            'ylabel': '# Unique State Bins',
            'color': '#1f77b4', 
            'desc': 'Grid: Pos × Vel'
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
            'desc': 'Crashes Only (MaxPos×AvgSpeed)'
        }
    ]

    for ax, config in zip(axes, metrics_config):
        data = history[config['key']]
        ax.plot(episodes, data, color=config['color'], linewidth=2.5, label='MDPFuzz')
        ax.fill_between(episodes, data, color=config['color'], alpha=0.1)
        
        ax.set_title(config['title'], fontweight='bold', pad=15)
        ax.set_xlabel('Episodes')
        ax.set_ylabel(config['ylabel'])
        
        ax.text(0.05, 0.95, config['desc'], transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.show()

def main():
    print(f"Target Obs File: {OBS_FILE}")
    parser = MdpFuzzLogParser(OBS_FILE)
    
    analyzer = DiversityAnalyzer(
        state_grid_size=(50, 50),
        behavior_grid_size=(50, 50)
    )
    
    history = analyzer.calculate_trends(parser)
    
    plot_metrics(history, save_path=PLOT_FILE)

if __name__ == "__main__":
    main()