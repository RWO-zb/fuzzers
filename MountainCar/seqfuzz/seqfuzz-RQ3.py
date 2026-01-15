import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 配置 ---
LOG_FILE = 'all_run_seeds_0.pkl'
MAX_H = 12.0 
SAVE_NAME = 'SeqFuzz_RQ3_Metrics.png'

# --- 绘图样式 ---
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
    if not os.path.exists(filepath):
        return 0, np.nan

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # 1. 筛选 Crashes
    crashes = []
    for entry in data:
        # enjoy.py 字段名为 'crashed'
        if not entry.get('crashed', False):
            continue
        crashes.append(entry)

    # 按时间排序
    crashes.sort(key=lambda x: x.get('crash_time', 0) if x.get('crash_time') is not None else float('inf'))

    unique_crashes = []
    seen_states = set()
    
    for c in crashes:
        state = c.get('state') # 字段名为 'state'
        if state is None: continue
        
        t = c.get('crash_time')
        if t is not None and t > MAX_H * 3600:
            continue
            
        try:
            if hasattr(state, 'tobytes'): state_bytes = state.tobytes()
            else: state_bytes = bytes(state)
        except: continue
        
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            unique_crashes.append(c)
            
    n_crashes = len(unique_crashes)
    print(f"Found {n_crashes} unique crashes.")
    
    # 指标 1: 时间效率 (秒/Crash)
    if n_crashes > 0:
        time_eff = (MAX_H * 3600) / n_crashes
    else:
        time_eff = 0
        
    # 指标 2: 平均代数
    # enjoy.py 直接保存了 'generation'，无需使用 parent_depth 计算
    if n_crashes > 0:
        generations = []
        for c in unique_crashes:
            gen = c.get('generation')
            if gen is not None:
                generations.append(gen)
        
        if generations:
            avg_gen = np.mean(generations)
        else:
            avg_gen = np.nan
    else:
        avg_gen = np.nan
        
    return time_eff, avg_gen

# --- 主逻辑 ---
data_sources = ["SeqFuzz"] 
files_map = {"SeqFuzz": LOG_FILE}

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

# 图 1: 时间效率
ax1 = axes[0]
bars1 = ax1.bar(metrics_data["labels"], metrics_data["time_per_crash"], 
                color=colors, alpha=0.8, edgecolor='black', width=0.5)
ax1.set_title("Time Cost Efficiency")
ax1.set_ylabel("Avg. seconds per Crash")
ax1.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars1:
    height = bar.get_height()
    label_text = f'{height:.1f} s' if height > 0 else 'N/A'
    ax1.text(bar.get_x() + bar.get_width()/2., height, label_text,
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 图 2: 平均代数
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
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'No Crash Data', ha='center', va='center', transform=ax2.transAxes)

ax2.set_title("Average Discovery Generation")
ax2.set_ylabel("Avg. Generation Index")
ax2.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(SAVE_NAME, dpi=300)
print(f"Plot saved to {SAVE_NAME}")
plt.show()