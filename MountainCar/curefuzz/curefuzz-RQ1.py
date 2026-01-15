import pickle
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = 'selection_log.pkl'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2
})

MAX_H = 12.0        
VIEW_LIMIT_H = 12.5 

def load_and_process_data(filepath): 
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    crashes = [entry for entry in data if entry.get('did_crash', False)]
    crashes.sort(key=lambda x: x.get('crash_time', 0) if x.get('crash_time') is not None else float('inf'))
    unique_crashes = []
    seen_states = set()
    
    for c in crashes:
        state = c.get('mutate_state')
        if state is None: continue
        
        state_bytes = state.tobytes()
        t = c.get('crash_time')
        if t is not None and t > MAX_H * 3600:
            continue
            
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            unique_crashes.append(c)
            
    times = np.array([c['crash_time'] for c in unique_crashes if c.get('crash_time') is not None])
    return times

data_sources = {
    "CureFuzz": {
        "file": LOG_FILE,
        "color": "#1f77b4" # 默认蓝色
    }
}

plt.figure(figsize=(10, 6))
markers_x_h = np.arange(2, MAX_H + 0.1, 2) 
for label, config in data_sources.items():
    times = load_and_process_data(config["file"])
    
    if len(times) == 0:
        print(f"No valid crash data found for {label}")
        continue
        
    times_h = times / 3600.0
    
    x_plot = np.concatenate(([0], times_h))
    y_plot = np.concatenate(([0], np.arange(1, len(times_h) + 1)))
    
    if x_plot[-1] < MAX_H:
        x_plot = np.concatenate((x_plot, [MAX_H]))
        y_plot = np.concatenate((y_plot, [y_plot[-1]]))
    
    line, = plt.step(x_plot, y_plot, where='post', label=label, color=config.get('color'))
    color = line.get_color()
    
    marker_y_vals = []
    for mx in markers_x_h:
        count = np.searchsorted(times_h, mx, side='right')
        marker_y_vals.append(count)
        
    plt.plot(markers_x_h, marker_y_vals, linestyle='none', marker='^', 
             color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)

plt.xlim(0, VIEW_LIMIT_H)
plt.xticks(np.arange(0, 13, 2))
plt.xlabel("Time (h)")
plt.ylabel("Number of Unique Crashes")
plt.title("Cumulative Unique Crashes (CureFuzz)")
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
save_name = 'CureFuzz_RQ1_Curve.png'
plt.savefig(save_name, dpi=300)
print(f"Plot saved to {save_name}")
plt.show()