import pickle
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = 'selection_log.pkl'
MAX_H = 12.0 

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 2
})

def calculate_metrics(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    crashes = [entry for entry in data if entry.get('did_crash', False)]
    crashes.sort(key=lambda x: x.get('crash_time', 0) if x.get('crash_time') is not None else float('inf'))

    unique_crashes = []
    seen_states = set()
    
    for c in crashes:
        state = c.get('mutate_state')
        if state is None: continue
        
        t = c.get('crash_time')
        if t is not None and t > MAX_H * 3600:
            continue
            
        state_bytes = state.tobytes()
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            unique_crashes.append(c)
            
    n_crashes = len(unique_crashes)
    
    if n_crashes > 0:
        time_eff = (MAX_H * 3600) / n_crashes
    else:
        time_eff = 0
        
    if n_crashes > 0:
        generations = []
        for c in unique_crashes:
            p_depth = c.get('parent_depth')
            gen = (p_depth + 1) if p_depth is not None else 1
            generations.append(gen)
        avg_gen = np.mean(generations)
    else:
        avg_gen = np.nan
        
    return time_eff, avg_gen

data_sources = ["CureFuzz"] 
files_map = {"CureFuzz": LOG_FILE}

metrics_data = {
    "labels": [],
    "time_per_crash": [],
    "gen_avg_depth": []
}

for label in data_sources:
    fname = files_map.get(label)
    t_eff, g_eff = calculate_metrics(fname)
    
    metrics_data["labels"].append(label)
    metrics_data["time_per_crash"].append(t_eff)
    metrics_data["gen_avg_depth"].append(g_eff)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = plt.cm.tab10(np.arange(len(metrics_data["labels"])))

ax1 = axes[0]
bars1 = ax1.bar(metrics_data["labels"], metrics_data["time_per_crash"], 
                color=colors, alpha=0.8, edgecolor='black', width=0.5)
ax1.set_title("Time Cost Efficiency")
ax1.set_ylabel("Avg. seconds per Crash")
ax1.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars1:
    height = bar.get_height()
    label_text = f'{height:.1f} min' if height > 0 else 'N/A'
    ax1.text(bar.get_x() + bar.get_width()/2., height, label_text,
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2 = axes[1]
valid_indices = [i for i, x in enumerate(metrics_data["gen_avg_depth"]) if not np.isnan(x)]
if valid_indices:
    valid_labels = [metrics_data["labels"][i] for i in valid_indices]
    valid_values = [metrics_data["gen_avg_depth"][i] for i in valid_indices]
    valid_colors = [colors[i] for i in valid_indices]

    bars2 = ax2.bar(valid_labels, valid_values, 
                    color=valid_colors, alpha=0.8, edgecolor='black', width=0.5)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'No Crash Data', ha='center', va='center', transform=ax2.transAxes)

ax2.set_title("Average Discovery Generation")
ax2.set_ylabel("Avg. Generation Index")
ax2.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
save_name = 'CureFuzz_RQ3_Metrics.png'
plt.savefig(save_name, dpi=300)
print(f"Plot saved to {save_name}")
plt.show()