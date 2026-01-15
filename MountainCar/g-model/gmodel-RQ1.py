import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 请修改为您的实验结果路径

LOG_FILE =  "all_test_cases_log.pkl"
SAVE_NAME = 'GModel_RQ1_Curve.png'
# ===========================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12, 'axes.labelsize': 14,
    'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12, 'lines.linewidth': 2
})

MAX_H = 12.0        # 最大显示时间 (小时)

def load_and_process_data(filepath): 
    if not os.path.exists(filepath):
        print(f"Error: File not found {filepath}")
        return np.array([])

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # 筛选 Crash (根据 timestamp 排序)
    crashes = [entry for entry in data if entry.get('is_crash', False)]
    crashes.sort(key=lambda x: x.get('timestamp', 0))
    
    unique_crashes = []
    seen_inputs = set()
    
    for c in crashes:
        inp = c.get('input')
        if inp is None: continue
        
        # 将 input 转为 tuple 以便去重
        inp_tuple = tuple(inp) if isinstance(inp, list) else tuple(inp.tolist())
        
        t = c.get('timestamp')
        if t is not None and t > MAX_H * 3600:
            continue
            
        if inp_tuple not in seen_inputs:
            seen_inputs.add(inp_tuple)
            unique_crashes.append(c)
            
    # 提取时间戳
    times = np.array([c['timestamp'] for c in unique_crashes if c.get('timestamp') is not None])
    return times

def main():
    data_sources = {"G-Model": {"file": LOG_FILE, "color": "#1f77b4"}} # 蓝色

    plt.figure(figsize=(10, 6))
    markers_x_h = np.arange(2, MAX_H + 0.1, 2) 
    
    for label, config in data_sources.items():
        times = load_and_process_data(config["file"])
        if len(times) == 0:
            print(f"No valid crash data found for {label}")
            continue
            
        times_h = times / 3600.0
        
        # 构造阶梯图数据
        x_plot = np.concatenate(([0], times_h))
        y_plot = np.concatenate(([0], np.arange(1, len(times_h) + 1)))
        
        if x_plot[-1] < MAX_H:
            x_plot = np.concatenate((x_plot, [MAX_H]))
            y_plot = np.concatenate((y_plot, [y_plot[-1]]))
        
        line, = plt.step(x_plot, y_plot, where='post', label=label, color=config.get('color'))
        color = line.get_color()
        
        # 添加标记点
        marker_y_vals = []
        for mx in markers_x_h:
            count = np.searchsorted(times_h, mx, side='right')
            marker_y_vals.append(count)
            
        plt.plot(markers_x_h, marker_y_vals, linestyle='none', marker='^', 
                 color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)

    plt.xlim(0, 12.5)
    plt.xticks(np.arange(0, 13, 2))
    plt.xlabel("Time (h)")
    plt.ylabel("Number of Unique Crashes")
    plt.title("Cumulative Unique Crashes (G-Model)")
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path =SAVE_NAME
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()