import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
import os

CSV_FILE = 'results/mc_test_data.csv'
PLOT_NAME = 'qdfuzz_RQ3.png'
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

def load_metrics_data(csv_path):
    print(f"Loading data from {csv_path} (Fuzzing Phase Only)...")
    if not os.path.exists(csv_path):
        print("Error: File not found.")
        return []
    df = pd.read_csv(csv_path)
    crashes = []
    for _, row in df.iterrows():
        m_count = int(row['mutation_count'])
        if m_count == 0:
            continue
        if not bool(row['is_faulty']):
            continue
        inp = row['input']
        if isinstance(inp, str):
            try:
                inp_val = np.array(json.loads(inp))
            except:
                try:
                    inp_val = np.array(ast.literal_eval(inp))
                except:
                    inp_val = inp
        else:
            inp_val = inp
        generation = m_count + 1
        entry = {
            'crash_time': row['discovery_time'],
            'mutate_state': inp_val,
            'generation': generation
        }
        crashes.append(entry)
    print(f"Loaded {len(crashes)} crashes from Fuzzing Phase.")
    return crashes

def plot_rq3(crashes):
    unique_crashes = []
    seen_states = set()
    crashes.sort(key=lambda x: x['crash_time'])
    for c in crashes:
        t = c['crash_time']
        if t > MAX_H * 3600: continue
        state = c['mutate_state']
        if hasattr(state, 'tobytes'):
            state_key = state.tobytes()
        else:
            state_key = str(state)
            
        if state_key not in seen_states:
            seen_states.add(state_key)
            unique_crashes.append(c)     
    n_crashes = len(unique_crashes)
    if n_crashes > 0:
        time_eff = (MAX_H * 3600) / n_crashes
    else:
        time_eff = 0
    if n_crashes > 0:
        generations = [c['generation'] for c in unique_crashes]
        avg_gen = np.mean(generations)
    else:
        avg_gen = np.nan    
    labels = ["QDFuzz (Fuzz Only)"]
    t_vals = [time_eff]
    g_vals = [avg_gen]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#1f77b4'] 
    ax1 = axes[0]
    bars1 = ax1.bar(labels, t_vals, color=colors, alpha=0.8, edgecolor='black', width=0.5)
    ax1.set_title("Time Cost Efficiency")
    ax1.set_ylabel("Avg. seconds per Crash")
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars1:
        height = bar.get_height()
        label_text = f'{height:.1f} s' if height > 0 else 'N/A'
        ax1.text(bar.get_x() + bar.get_width()/2., height, label_text,
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2 = axes[1]
    if not np.isnan(avg_gen):
        bars2 = ax2.bar(labels, g_vals, color=colors, alpha=0.8, edgecolor='black', width=0.5)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Crash Data', ha='center', va='center', transform=ax2.transAxes)

    ax2.set_title("Average Discovery Generation")
    ax2.set_ylabel("Avg. Generation Index")
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_NAME, dpi=300)
    print(f"Plot saved to {PLOT_NAME}")
    plt.show()

if __name__ == "__main__":
    data = load_metrics_data(CSV_FILE)
    if data:
        plot_rq3(data)