import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 配置 ---
LOG_FILE = 'all_run_seeds_0.pkl'
MAX_H = 12.0        
VIEW_LIMIT_H = 12.5 
SAVE_NAME = 'SeqFuzz_RQ1_Curve.png'

# --- 绘图样式 ---
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

def load_and_process_data(filepath): 
    if not os.path.exists(filepath):
        print(f"Error: File not found {filepath}")
        return []
        
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # 1. 筛选 Crash
    # 根据 enjoy.py，字段名为 'crashed' (boolean)
    crashes = []
    for entry in data:
        if entry.get('crashed', False):
            crashes.append(entry)
            
    # 2. 按时间排序
    crashes.sort(key=lambda x: x.get('crash_time', 0) if x.get('crash_time') is not None else float('inf'))
    
    unique_crashes = []
    seen_states = set()
    
    for c in crashes:
        # enjoy.py 中使用的是 'state'
        state = c.get('state')
        if state is None: continue
        
        # 3. 状态去重
        try:
            if hasattr(state, 'tobytes'):
                state_bytes = state.tobytes()
            else:
                state_bytes = bytes(state)
        except:
            continue

        t = c.get('crash_time')
        # 时间过滤
        if t is not None and t > MAX_H * 3600:
            continue
            
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            unique_crashes.append(c)
            
    # 提取时间戳
    times = np.array([c['crash_time'] for c in unique_crashes if c.get('crash_time') is not None])
    return times

# --- 主逻辑 ---
data_sources = {
    "SeqFuzz": {
        "file": LOG_FILE,
        "color": "#1f77b4" # 蓝色
    }
}

plt.figure(figsize=(10, 6))
markers_x_h = np.arange(2, MAX_H + 0.1, 2) 

for label, config in data_sources.items():
    times = load_and_process_data(config["file"])
    print(f"[{label}] 发现的 Unique Crashes 数量: {len(times)}")
    if len(times) == 0:
        print(f"No valid crash data found for {label}")
        # 如果没有数据，画一条空线
        x_plot = np.array([0, MAX_H])
        y_plot = np.array([0, 0])
        times_h = np.array([])
    else:
        times_h = times / 3600.0
        x_plot = np.concatenate(([0], times_h))
        y_plot = np.concatenate(([0], np.arange(1, len(times_h) + 1)))
        
        # 延伸至最大时间
        if x_plot[-1] < MAX_H:
            x_plot = np.concatenate((x_plot, [MAX_H]))
            y_plot = np.concatenate((y_plot, [y_plot[-1]]))
    
    line, = plt.step(x_plot, y_plot, where='post', label=label, color=config.get('color'))
    color = line.get_color()
    
    # 添加三角形标记
    if len(times_h) > 0:
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
plt.title("Cumulative Unique Crashes (SeqFuzz)")
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(SAVE_NAME, dpi=300)
print(f"Plot saved to {SAVE_NAME}")
plt.show()