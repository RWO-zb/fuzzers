import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

LOG_FILE = 'selection_log.pkl'
PLOT_BD_CURVE = 'behaviour_diversity_curve.png'
PLOT_FD_CURVE = 'fault_diversity_curve.png'
PLOT_SC_CURVE = 'state_coverage_curve.png'

GRID_SIZE = (50, 50)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(f"Loaded {len(data)} entries from {file_path}")
        return data
    
def get_bin_index(value, min_val, max_val, grid_size_dim):
    if max_val <= min_val:
        return 0
    idx = int((value - min_val) / (max_val - min_val) * grid_size_dim)
    return min(max(idx, 0), grid_size_dim - 1)

def calculate_cumulative_trends(log_data):
    all_dists = []
    all_angles = []
    
    for entry in log_data:
        d = entry.get('bd_distance')
        a = entry.get('bd_mean_angle')
        if d is not None and a is not None:
            all_dists.append(d)
            all_angles.append(a)

    min_dist, max_dist = min(all_dists), max(all_dists) + 1e-5
    min_angle, max_angle = min(all_angles), max(all_angles) + 1e-5
    
    print(f"Global Range - Dist: [{min_dist:.2f}, {max_dist:.2f}], Angle: [{min_angle:.2f}, {max_angle:.2f}]")

    bd_filled_bins = set()     
    fd_crash_bins = set()      
    unique_states = set()    
    
    bd_trend = []
    fd_trend = []
    sc_trend = []
    
    for i, entry in enumerate(log_data):
        state = entry.get('mutate_state')
        if state is not None:  
            state_bytes = state.tobytes() if hasattr(state, 'tobytes') else np.array(state).tobytes()
            unique_states.add(state_bytes)
        sc_trend.append(len(unique_states))

        d = entry.get('bd_distance')
        a = entry.get('bd_mean_angle')
        is_crash = entry.get('did_crash', False)
        
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
        'sc_trend': sc_trend
    }

def plot_curve(x, y, title, ylabel, filename, color):
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

    plot_curve(
        x, trends['bd_trend'], 
        title='Behaviour Diversity Growth', 
        ylabel='Cumulative Covered Bins (Behaviour)', 
        filename=PLOT_BD_CURVE,
        color='teal'
    )
    
    plot_curve(
        x, trends['fd_trend'], 
        title='Fault Diversity Growth', 
        ylabel='Cumulative Covered Crash Bins', 
        filename=PLOT_FD_CURVE,
        color='crimson'
    )
    
    plot_curve(
        x, trends['sc_trend'], 
        title='State Coverage Growth', 
        ylabel='Cumulative Unique Mutate States', 
        filename=PLOT_SC_CURVE,
        color='royalblue'
    )

    print("\nAll curves generated successfully.")

if __name__ == "__main__":
    main()