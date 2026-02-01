import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


MAX_SEEDS = 1000         
OUTPUT_FILE = 'RQ2-CARLA_Final_Counts_SuccessFalse.png'


TOTAL_GRID_SIZE = 10000  

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

files_config = {
    "curefuzz.csv": {
        "label": "CureFuzz", 
        "phase_col": "phase", 
        "target_phase": "Phase2",
        "color": "#1f77b4"  
    },
    "g-model.csv": {
        "label": "G-Model", 
        "special": "direct", 
        "color": "#ff7f0e"  
    },
    "mdpfuzz.csv": {
        "label": "MDPFuzz", 
        "phase_col": "phase", 
        "target_phase": "Phase2",
        "color": "#2ca02c"  
    },
    "qdfuzz.csv": {
        "label": "QDFuzz", 
        "phase_col": "phase", 
        "target_phase": "Phase2",
        "color": "#d62728" 
    },
    "random.csv": {
        "label": "Random", 
        "special": "direct", 
        "color": "#9467bd"  
    },
    "seqfuzz.csv": {
        "label": "SeqFuzz", 
        "phase_col": "phase", 
        "target_phase": "Phase2",
        "color": "#8c564b"  
    }
}


metrics = [
    {"col": "state_coverage", "title": "State Coverage Growth", "ylabel": "# Unique States"},
    {"col": "behavior_count", "title": "Behavior Diversity Growth", "ylabel": "# Unique Behaviors"},
    {"col": "fault_behavior_count", "title": "Fault Diversity Growth", "ylabel": "# Unique Faults"}
]


def print_final_stats(final_stats):
    print("\n" + "="*95)
    print(f"{'Method':<15} | {'Unique States':<15} | {'Behav Div':<12} | {'Fault Div':<12} | {'Crash Seeds':<12}")
    print("-" * 95)
    
    sorted_names = sorted(final_stats.keys())
    
    for name in sorted_names:
        stats = final_stats[name]
        sc = stats.get('state_coverage', 0)
        bd = stats.get('behavior_count', 0)
        fd = stats.get('fault_behavior_count', 0)
        cs = stats.get('crash_seeds', 0)
        
        print(f"{name:<15} | {int(sc):<15} | {int(bd):<12} | {int(fd):<12} | {int(cs):<12}")
    
    print("="*95 + "\n")

def main():
    data_store = {m["col"]: {} for m in metrics}
    final_stats = {}

    print("--- Loading Data ---")

    for fname, config in files_config.items():
        label = config["label"]
        if label not in final_stats:
            final_stats[label] = {}
            
        try:
            if os.path.exists(fname):
                df = pd.read_csv(fname)
            elif os.path.exists(f"plot/{fname}"):
                df = pd.read_csv(f"plot/{fname}")
            else:
                print(f"[Warning] File not found: {fname}, skipping.")
                continue
            
            df_filtered = pd.DataFrame()

            if config.get("special") == "direct":
                df_filtered = df.copy()

            else:
                target_phase = config["target_phase"]
                phase_col = config.get("phase_col", "phase")
                
                if phase_col in df.columns:
                    if target_phase in df[phase_col].values:
                        df_filtered = df[df[phase_col] == target_phase].copy()
                    else:
                        print(f"[Info] {label}: '{target_phase}' not found, using all data.")
                        df_filtered = df.copy()
                else:
                    df_filtered = df.copy()

            if len(df_filtered) > MAX_SEEDS:
                df_filtered = df_filtered.iloc[:MAX_SEEDS]
            
            df_filtered = df_filtered.reset_index(drop=True)
            
            crash_count = 0
            if 'success' in df_filtered.columns:
                crashes = df_filtered[df_filtered['success'] == False]
                
                if 'weather_id' in crashes.columns:
                    seed_cols = ['weather_id', 'start_id', 'target_id']
                elif 'weather' in crashes.columns:
                    seed_cols = ['weather', 'start_id', 'target_id']
                else:
                    seed_cols = ['start_id', 'target_id']
                
                valid_cols = [c for c in seed_cols if c in crashes.columns]
                
                if len(crashes) > 0 and len(valid_cols) > 0:
                    unique_seeds = crashes[valid_cols].drop_duplicates()
                    crash_count = len(unique_seeds)
            
            final_stats[label]['crash_seeds'] = crash_count

            for m in metrics:
                col_name = m["col"]
                if col_name in df_filtered.columns:
                    values = df_filtered[col_name].values
                    
                    if col_name == "state_coverage":
                        if len(values) > 0 and np.max(values) <= 1.0:
                            values = values * TOTAL_GRID_SIZE
                    
                    data_store[col_name][label] = values
                    
                    if len(values) > 0:
                        final_stats[label][col_name] = values[-1]
                    else:
                        final_stats[label][col_name] = 0

            print(f"Loaded {label}: {len(df_filtered)} seeds. (Unique 'Success=False' Seeds: {crash_count})")

        except Exception as e:
            print(f"[Error] Processing {fname}: {e}")

    print_final_stats(final_stats)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    for i, m in enumerate(metrics):
        ax = axes[i]
        col_name = m["col"]
        
        for fname, config in files_config.items():
            label = config["label"]
            if label in data_store[col_name]:
                values = data_store[col_name][label]
                x_axis = range(1, len(values) + 1)
                ax.plot(x_axis, values, label=label, color=config.get("color"), alpha=0.9, linewidth=2)
                
        ax.set_title(m["title"], fontweight='bold', fontsize=14)
        ax.set_xlabel("Number of Test Cases", fontsize=12)
        ax.set_ylabel(m["ylabel"], fontsize=12)
        ax.set_xlim(left=0, right=MAX_SEEDS)
        ax.set_ylim(bottom=0) 
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=10)

    ax_bar = axes[3]
    
    bar_labels = []
    bar_counts = []
    bar_colors = []
    
    for fname, config in files_config.items():
        label = config["label"]
        if label in final_stats:
            count = final_stats[label].get('crash_seeds', 0)
            bar_labels.append(label)
            bar_counts.append(count)
            bar_colors.append(config.get("color", "gray"))
    
    x_pos = range(len(bar_labels))
    bars = ax_bar.bar(x_pos, bar_counts, color=bar_colors, alpha=0.9, width=0.6)
    
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=11)
    
    ax_bar.set_title("Distinct Crash Seeds\n(Success = False)", fontweight='bold', fontsize=14)
    ax_bar.set_ylabel("# Initial Seeds Causing Crash", fontsize=12)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Plot saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()