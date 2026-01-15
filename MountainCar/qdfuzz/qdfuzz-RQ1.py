import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
import os

CSV_FILE = 'results/mc_test_data.csv'
PLOT_NAME = 'qdfuzz_RQ1.png'

MAX_H = 12.0        
VIEW_LIMIT_H = 12.5 

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

def load_crash_data(csv_path):
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return []
    df = pd.read_csv(csv_path)
    init_df = df[df['mutation_count'] == 0]
    if not init_df.empty:
        fuzz_start_offset = init_df['discovery_time'].max()
        print(f"Initialization phase detected. Subtracting {fuzz_start_offset:.2f}s from timelines.")
    else:
        fuzz_start_offset = 0.0
        print("No initialization phase data found. Using absolute time.")
    crashes = []
    for _, row in df.iterrows():
        if int(row['mutation_count']) == 0:
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

        raw_time = float(row['discovery_time'])
        fuzz_time = raw_time - fuzz_start_offset
        fuzz_time = max(0.0, fuzz_time)
        entry = {
            'crash_time': fuzz_time,
            'mutate_state': inp_val
        }
        crashes.append(entry)
    print(f"Loaded {len(crashes)} valid crashes from Fuzzing Phase.")
    return crashes

def plot_rq1(crashes):
    crashes.sort(key=lambda x: x.get('crash_time', 0)) 
    unique_crashes = []
    seen_states = set()
    for c in crashes:
        t = c.get('crash_time')
        if t is not None and t > MAX_H * 3600:
            continue
        state = c.get('mutate_state')
        if hasattr(state, 'tobytes'):
            state_key = state.tobytes()
        else:
            state_key = str(state)

        if state_key not in seen_states:
            seen_states.add(state_key)
            unique_crashes.append(c)     
    times = np.array([c['crash_time'] for c in unique_crashes])
    plt.figure(figsize=(10, 6))
    if len(times) > 0:
        times_h = times / 3600.0
        x_plot = np.concatenate(([0], times_h))
        y_plot = np.concatenate(([0], np.arange(1, len(times_h) + 1)))
        if x_plot[-1] < MAX_H:
            x_plot = np.concatenate((x_plot, [MAX_H]))
            y_plot = np.concatenate((y_plot, [y_plot[-1]]))
        label = "QDFuzz "
        color = "#1f77b4"
        plt.step(x_plot, y_plot, where='post', label=label, color=color)
        markers_x_h = np.arange(2, MAX_H + 0.1, 2)
        marker_y_vals = []
        for mx in markers_x_h:
            count = np.searchsorted(times_h, mx, side='right')
            marker_y_vals.append(count)
        plt.plot(markers_x_h, marker_y_vals, linestyle='none', marker='^', 
                 color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)
    else:
        print("No unique crashes found in Fuzzing Phase within time budget.")

    plt.xlim(0, VIEW_LIMIT_H)
    plt.xticks(np.arange(0, 13, 2))
    plt.xlabel("Time in Fuzz Stage (h)") 
    plt.ylabel("Number of Unique Crashes")
    plt.title("Cumulative Unique Crashes ")
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_NAME, dpi=300)
    print(f"Plot saved to {PLOT_NAME}")
    plt.show()

if __name__ == "__main__":
    data = load_crash_data(CSV_FILE)
    if data:
        plot_rq1(data)