import matplotlib.pyplot as plt
import numpy as np
import csv

#LOG_FILE = 'logs/MC_DQN_NoCov_5_0.01_0.1_0_12h_logs.txt'
LOG_FILE = 'logs/MC_DQN_RT_1022_10000it_logs.txt' 

#LABEL = 'MDPFuzz' 
LABEL = 'Random Testing'

#IS_RT = False 
IS_RT = True
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

def load_fuzz_data(filepath, is_rt_mode=False):
    unique_crashes = []
    seen_inputs = set()
    fuzz_start_time = None  
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader, None)
        if not headers: return np.array([])
        headers = [h.strip() for h in headers]
        idx_input = headers.index('Input')
        idx_oracle = headers.index('Oracle')
        idx_runtime = headers.index('RunTime')
        idx_gen = headers.index('Generation') 
        rows = []
        for row in reader:
            if row: rows.append(row)
        rows.sort(key=lambda x: float(x[idx_runtime]) if x[idx_runtime].strip() != 'None' else 0)
        for row in rows:
            if len(row) <= idx_gen: continue
            gen_val = int(float(row[idx_gen]))
            if not is_rt_mode and gen_val == 0:
                continue 
            run_time = float(row[idx_runtime])
            if fuzz_start_time is None:
                    fuzz_start_time = run_time       
            relative_time = run_time - fuzz_start_time   
            if relative_time > MAX_H * 3600:
                continue
            oracle_str = row[idx_oracle].strip()
            inp_str = row[idx_input].strip() 
            if oracle_str == 'True':
                if inp_str not in seen_inputs:
                    seen_inputs.add(inp_str)
                    unique_crashes.append(relative_time)            
    return np.array(unique_crashes)

plt.figure(figsize=(10, 6))
markers_x_h = np.arange(2, MAX_H + 0.1, 2)

times = load_fuzz_data(LOG_FILE, is_rt_mode=IS_RT)

if len(times) == 0:
    print(f"[{LABEL}] No crashes found in Fuzzing Phase.")
    times_h = np.array([0])
    x_plot, y_plot = np.array([0]), np.array([0])
else:
    times_h = times / 3600.0
    x_plot = np.concatenate(([0], times_h))
    y_plot = np.concatenate(([0], np.arange(1, len(times_h) + 1)))
    if x_plot[-1] < MAX_H:
        x_plot = np.concatenate((x_plot, [MAX_H]))
        y_plot = np.concatenate((y_plot, [y_plot[-1]]))

plot_color = '#1f77b4' if IS_RT else '#ff7f0e' 

line, = plt.step(x_plot, y_plot, where='post', label=LABEL, color=plot_color)

if len(times) > 0:
    marker_y_vals = [np.searchsorted(times_h, mx, side='right') for mx in markers_x_h]
    plt.plot(markers_x_h, marker_y_vals, linestyle='none', marker='^', 
             color=plot_color, markersize=8, markeredgecolor='white', markeredgewidth=1)

plt.xlim(0, VIEW_LIMIT_H)
plt.xticks(np.arange(0, 13, 2))
plt.xlabel("Fuzzing Time (h) ") 
plt.ylabel("Number of Unique Crashes")
plt.title(f"Cumulative Unique Crashes ({LABEL})")
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
save_name = 'RQ1_RT.png' if IS_RT else 'RQ1_MDPFuzz.png'
plt.savefig(save_name, dpi=300)
print(f"Saved plot to {save_name}")
plt.show()