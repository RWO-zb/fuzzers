import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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

files_config = {
    "curefuzz.csv": {
        "label": "CureFuzz", 
        "time_col": "elapsed_time", 
        "gen_col": "mutation_generation",
        "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"
    },
    "g-model.csv": {
        "label": "G-Model", 
        "time_col": "elapsed_time", 
        "gen_col": None, 
        "special": "g-model", "input_col": "input_post"
    },
    "mdpfuzz.csv": {
        "label": "MDPFuzz", 
        "time_col": "global_time", 
        "gen_col": "generation",
        "phase_col": "phase", "target_phase": "Phase2", "input_col": "current_input"
    },
    "qdfuzz.csv": {
        "label": "QDFuzz", 
        "time_col": "elapsed_time", 
        "gen_col": "mutation_generation",
        "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"
    },
    "random.csv": {
        "label": "Random", 
        "time_col": "global_time", 
        "gen_col": None, 
        "special": "random", "input_col": "current_input"
    },
    "seqfuzz.csv": {
        "label": "SeqFuzz", 
        "time_col": "elapsed_time", 
        "gen_col": "mutation_generation",
        "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"
    }
}


max_h = 12.0 
metrics_data = {
    "labels": [],
    "time_per_crash": [], 
    "gen_avg_depth": []   
}

gen_distributions = {} 


for fname, config in files_config.items():
    try:
        try:
            df = pd.read_csv(fname)
        except FileNotFoundError:
            print(f"Warning: File {fname} not found. Skipping.")
            continue

        label = config["label"]
        metrics_data["labels"].append(label)
        
        if config.get("special") == "g-model":
          
            df = df.reset_index(drop=True)
            df['generation'] = (df.index // 20) + 1
           
            config['gen_col'] = 'generation'

        if "special" in config and config["special"] == "g-model":
            start_time = df[config['time_col']].min()
            df_filtered = df.copy()
            
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
                metrics_data["time_per_crash"].append(0)
                metrics_data["gen_avg_depth"].append(np.nan)
                continue

      
        df_filtered['norm_time'] = df_filtered[config['time_col']] - start_time
        
      
        limit_sec = max_h * 3600
        df_12h = df_filtered[df_filtered['norm_time'] <= limit_sec].copy()
        
       
        if df_12h['success'].dtype == 'bool':
             is_crash = df_12h['success'] == False
        else:
             is_crash = df_12h['success'].astype(str) == 'False'
        
        crashes = df_12h[is_crash].copy()
        input_col = config.get("input_col")
        
        if not crashes.empty and input_col in crashes.columns:
          
            crashes = crashes.sort_values('norm_time')
            
            unique_crashes = crashes.drop_duplicates(subset=[input_col], keep='first')
            n_crashes = len(unique_crashes)
        else:
            unique_crashes = pd.DataFrame()
            n_crashes = 0
            
      
        if n_crashes > 0:
            time_eff = (max_h * 60) / n_crashes 
        else:
            time_eff = 0 
            
        metrics_data["time_per_crash"].append(time_eff)
        
        
        gen_col = config.get("gen_col")
        
        if gen_col and gen_col in unique_crashes.columns and n_crashes > 0:
        
            all_gens = unique_crashes[gen_col].dropna()
            
           
            valid_gens_list = all_gens[all_gens > 0].tolist()
            if valid_gens_list:
                gen_distributions[label] = valid_gens_list
            
            avg_gen = all_gens.mean()
            metrics_data["gen_avg_depth"].append(avg_gen)
        else:
            metrics_data["gen_avg_depth"].append(np.nan)
            
    except Exception as e:
        print(f"Error processing {fname}: {e}")

all_labels = metrics_data["labels"]
palette = plt.cm.tab10(np.arange(len(all_labels)))
color_map = dict(zip(all_labels, palette)) 

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
bar_colors = [color_map[l] for l in metrics_data["labels"]]
bars1 = ax1.bar(metrics_data["labels"], metrics_data["time_per_crash"], 
                color=bar_colors, alpha=0.8, edgecolor='black')
ax1.set_title("Time Cost Efficiency")
ax1.set_ylabel("Avg. Minutes per Crash")
ax1.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars1:
    height = bar.get_height()
    label_text = f'{height:.1f} min' if height > 0 else 'N/A'
    ax1.text(bar.get_x() + bar.get_width()/2., height, label_text,
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2 = axes[1]

valid_indices = [i for i, x in enumerate(metrics_data["gen_avg_depth"]) if not np.isnan(x)]
valid_labels = [metrics_data["labels"][i] for i in valid_indices]
valid_values = [metrics_data["gen_avg_depth"][i] for i in valid_indices]
valid_colors = [color_map[l] for l in valid_labels]

bars2 = ax2.bar(valid_labels, valid_values, color=valid_colors, alpha=0.8, edgecolor='black')
ax2.set_title("Average Discovery Generation")
ax2.set_ylabel("Avg. Generation Index")
ax2.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('RQ3-BarCharts.png', dpi=300)
print("Saved Bar Charts to RQ3-BarCharts.png")

if gen_distributions:
    sorted_labels = [l for l in metrics_data["labels"] if l in gen_distributions]
    plot_data = [gen_distributions[l] for l in sorted_labels]
    plot_colors = [color_map[l] for l in sorted_labels]

    fig_box, ax_box = plt.subplots(figsize=(10, 6))

    box = ax_box.boxplot(plot_data, vert=False, patch_artist=True,
                         labels=sorted_labels, showmeans=True,
                         widths=0.6,
                         meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":8},
                         medianprops={"color": "black", "linewidth": 1.5})


    for patch, color in zip(box['boxes'], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

  
    for i, (method, values) in enumerate(zip(sorted_labels, plot_data)):
        y_pos = i + 1
        y_jitter = np.random.normal(y_pos, 0.08, size=len(values))
        ax_box.scatter(values, y_jitter, alpha=0.6, color=color_map[method], 
                       s=15, edgecolor='white', linewidth=0.5)

    ax_box.set_title('Distribution of Unique Crashes by Generation', fontsize=16, fontweight='bold')
    ax_box.set_xlabel('Generation Number (Log Scale)', fontsize=14)
    
    ax_box.set_xscale('symlog', linthresh=1)
    ax_box.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    ax_box.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig('RQ3-Boxplot.png', dpi=300)
    print("Saved Boxplot to RQ3-Boxplot.png")
    plt.show()
else:
    print("No generation data available for Boxplot.")