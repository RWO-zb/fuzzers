import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

files_config = {
    "curefuzz.csv": {
        "label": "CureFuzz", 
        "time_col": "elapsed_time", 
        "phase_col": "phase", 
        "target_phase": "Phase2", 
        "input_col": "input_post"
    },
    "g-model.csv": {
        "label": "G-Model", 
        "time_col": "elapsed_time", 
        "special": "g-model", 
        "input_col": "input_post"
    },
    "mdpfuzz.csv": {
        "label": "MDPFuzz", 
        "time_col": "global_time", 
        "phase_col": "phase", 
        "target_phase": "Phase2", 
        "input_col": "current_input"
    },
    "qdfuzz.csv": {
        "label": "QDFuzz", 
        "time_col": "elapsed_time", 
        "phase_col": "phase", 
        "target_phase": "Phase2", 
        "input_col": "input_post"
    },
    "random.csv": {
        "label": "Random", 
        "time_col": "global_time", 
        "special": "random", 
        "input_col": "current_input"
    },
    "seqfuzz.csv": {
        "label": "SeqFuzz", 
        "time_col": "elapsed_time", 
        "phase_col": "phase", 
        "target_phase": "Phase2", 
        "input_col": "input_post"
    }
}

data_store = {}
max_h = 12.0       
view_limit_h = 12.5 


for fname, config in files_config.items():
    try:
        df = pd.read_csv(fname)
        
        if "special" in config and config["special"] == "g-model":
            if 'generative+novelty' in df['method'].values:
                start_time = df[df['method'] == 'generative+novelty'][config['time_col']].min()
            else:
                start_time = df[config['time_col']].min()
            df_filtered = df[df[config['time_col']] >= start_time].copy()
            
        elif "special" in config and config["special"] == "random":
            start_time = df[config['time_col']].min()
            df_filtered = df.copy()
            
        else:
            target_phase = config["target_phase"]
            if target_phase in df[config['phase_col']].values:
                phase_data = df[df[config['phase_col']] == target_phase]
                start_time = phase_data[config['time_col']].min()
                df_filtered = phase_data.copy()
            else:
                print(f"Warning: {target_phase} not found in {fname}")
                continue
                
    
        t_col = config["time_col"]
        df_filtered['norm_time'] = df_filtered[t_col] - start_time
        

        if df_filtered['success'].dtype == 'bool':
             is_crash = df_filtered['success'] == False
        else:
             is_crash = df_filtered['success'].astype(str) == 'False'
        
        crashes = df_filtered[is_crash].copy()
        
        limit_sec = max_h * 3600
        crashes = crashes[crashes['norm_time'] <= limit_sec]

        input_col = config.get("input_col")
        if input_col and input_col in crashes.columns:
            crashes = crashes.sort_values('norm_time')
            crashes = crashes.drop_duplicates(subset=[input_col], keep='first')
        else:
            print(f"Warning: Input column {input_col} not found in {fname}")
            
        times = np.sort(crashes['norm_time'].values)
        data_store[fname] = times
        
    except Exception as e:
        print(f"Error processing {fname}: {e}")


plt.figure(figsize=(10, 6))

markers_x_h = np.arange(2, max_h + 0.1, 2)

for fname, config in files_config.items():
    label = config["label"]
    times = data_store.get(fname, np.array([]))
    
    times_h = times / 3600.0
    
   
    x_plot = np.concatenate(([0], times_h))
    y_plot = np.concatenate(([0], np.arange(1, len(times_h) + 1)))
    
    if x_plot[-1] < max_h:
        x_plot = np.concatenate((x_plot, [max_h]))
        y_plot = np.concatenate((y_plot, [y_plot[-1]]))
    
    line, = plt.step(x_plot, y_plot, where='post', label=label)
    color = line.get_color()
    
    marker_y_vals = []
    for mx in markers_x_h:
        count = np.searchsorted(times_h, mx, side='right')
        marker_y_vals.append(count)
        
    plt.plot(markers_x_h, marker_y_vals, linestyle='none', marker='^', 
             color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)

plt.xlim(0, view_limit_h) 
plt.xticks(np.arange(0, 13, 2)) 
plt.xlabel("Time (h)")
plt.ylabel("Number of Unique Crashes")
plt.title("Cumulative Unique Crashes (CARLA)")
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('RQ1-CARLA.png', dpi=300)
plt.show()