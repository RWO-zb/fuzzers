import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

OBS_FILE = 'obs_sequences.pkl'
LOG_FILE = 'selection_log.pkl'
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
    def __init__(self, obs_path, log_path, 
                 state_grid_size=(100, 100), 
                 behavior_grid_size=(100, 100)):
        
        self.obs_path = obs_path
        self.log_path = log_path
        self.state_grid_size = state_grid_size
        self.behavior_grid_size = behavior_grid_size
        
        self.state_pos_range = (-1.2, 0.6)
        self.state_vel_range = (-0.07, 0.07)
        
        self.bd_max_pos_range = (-1.2, 0.6)
        self.bd_avg_speed_range = (0.0, 0.05)
        
        self.obs_seqs = self._load_pickle(obs_path)
        self.logs = self._load_pickle(log_path)

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

    def calculate_trends(self):
        if not self.obs_seqs or not self.logs:
            print("No data loaded! Please check your file paths.")
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

    plt.tight_layout()
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.show()

def main():
    print(f"Target Obs File: {OBS_FILE}")
    print(f"Target Log File: {LOG_FILE}")

    analyzer = DiversityAnalyzer(
        obs_path=OBS_FILE, 
        log_path=LOG_FILE,
        state_grid_size=(50, 50),    
        behavior_grid_size=(50, 50)   
    )
    
    history_data = analyzer.calculate_trends()
    
    plot_metrics(history_data, save_path=PLOT_FILE)

if __name__ == "__main__":
    main()