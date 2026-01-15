import matplotlib.pyplot as plt
import numpy as np
import csv

#LOG_FILE = 'logs/MC_DQN_NoCov_5_0.01_0.1_0_12h_logs.txt'
LOG_FILE = 'logs/MC_DQN_RT_0_budget730000_logs.txt'
#LABEL = 'MDPFuzz'
LABEL = 'Random Testing'
#IS_RT = False 
IS_RT = True
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
def calculate_metrics(filepath, is_rt_mode=False):
    unique_crashes = []
    seen_inputs = set()
    fuzz_start_time = None
    with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            headers = next(reader, None)
            if not headers: return 0, np.nan
            headers = [h.strip() for h in headers]
            idx_input = headers.index('Input')
            idx_oracle = headers.index('Oracle')
            idx_gen = headers.index('Generation')
            idx_runtime = headers.index('RunTime')
            rows = list(reader)
            rows.sort(key=lambda x: float(x[idx_runtime]) if x[idx_runtime].strip() != 'None' else 0)
            for row in rows:
                if not row: continue
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
                        unique_crashes.append({
                            'time': relative_time,
                            'generation': gen_val
                        })    
    n_crashes = len(unique_crashes)
    if n_crashes > 0:
        time_eff = (MAX_H * 3600) / n_crashes
    else:
        time_eff = 0
    if n_crashes > 0:
        if is_rt_mode:
            avg_gen = 0 
        else:
            avg_gen = np.mean([c['generation'] for c in unique_crashes])
    else:
        avg_gen = 0 
    return time_eff, avg_gen

t_eff, g_eff = calculate_metrics(LOG_FILE, is_rt_mode=IS_RT)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_color = '#1f77b4' if IS_RT else '#ff7f0e'
ax1 = axes[0]
bars1 = ax1.bar([LABEL], [t_eff], color=plot_color, alpha=0.8, edgecolor='black', width=0.4)
ax1.set_title("Time Cost Efficiency")
ax1.set_ylabel("Avg. Seconds per Crash")
ax1.grid(axis='y', linestyle='--', alpha=0.6)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
             f'{bar.get_height():.1f} s' if bar.get_height()>0 else 'N/A', ha='center', va='bottom')

ax2 = axes[1]
bars2 = ax2.bar([LABEL], [g_eff], color=plot_color, alpha=0.8, edgecolor='black', width=0.4)
ax2.set_title("Average Discovery Generation")
ax2.set_ylabel("Avg. Generation Index")
ax2.grid(axis='y', linestyle='--', alpha=0.6)
if IS_RT:
    ax2.text(0, 0.1, "N/A (Random Testing)", ha='center', va='bottom', color='black', fontweight='bold')
else:
    for bar in bars2:
        val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., val, 
                 f'{val:.1f}' if val > 0 else 'N/A', ha='center', va='bottom')
plt.tight_layout()
save_name = 'MDPFuzz_RQ3_Single.png'
plt.savefig(save_name, dpi=300)
print(f"Saved plot to {save_name}")
plt.show()