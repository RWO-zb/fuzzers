import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
LOG_FILE =  "all_test_cases_log.pkl"
SAVE_NAME = 'GModel_RQ3_Metrics.png'
# ===========================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12, 'axes.labelsize': 13,
    'axes.titlesize': 14, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'lines.linewidth': 2
})

def calculate_metrics(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return 0
        
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if not data:
        return 0

    # 1. 获取实验总时长 (取最后一条日志的时间戳)
    # 如果列表未按时间排序，先取 max
    total_duration_sec = max([entry.get('timestamp', 0) for entry in data])
    if total_duration_sec == 0:
        total_duration_sec = 1.0 # 避免除以零

    # 2. 筛选并去重 Crash
    crashes = [entry for entry in data if entry.get('is_crash', False)]
    unique_crashes = []
    seen_inputs = set()
    
    for c in crashes:
        inp = c.get('input')
        if inp is None: continue
        inp_tuple = tuple(inp) if isinstance(inp, list) else tuple(inp.tolist())
        
        if inp_tuple not in seen_inputs:
            seen_inputs.add(inp_tuple)
            unique_crashes.append(c)
            
    n_crashes = len(unique_crashes)
    
    # 3. 计算时间效率: 总时间 / 唯一Crash数量
    if n_crashes > 0:
        time_eff_sec = total_duration_sec / n_crashes
    else:
        time_eff_sec = 0
        
    print(f"Debug: Total Time={total_duration_sec:.2f}s, Unique Crashes={n_crashes}, Eff={time_eff_sec:.4f}s")
    return time_eff_sec

def main():
    data_sources = ["G-Model"] 
    files_map = {"G-Model": LOG_FILE}
    
    labels = []
    values = []

    for label in data_sources:
        fname = files_map.get(label)
        t_eff = calculate_metrics(fname)
        labels.append(label)
        values.append(t_eff)

    # 绘图
    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10(np.arange(len(labels)))

    bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', width=0.5)
    
    plt.title("Time Cost Efficiency")
    plt.ylabel("Avg. seconds per Crash")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # --- 关键修改：智能标签显示 ---
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            if height < 60:
                # 小于60秒，显示秒 (如 1.0 s)
                label_text = f'{height:.2f} s'
            else:
                # 大于60秒，显示分钟 (如 1.5 min)
                label_text = f'{height/60.0:.1f} min'
        else:
            label_text = 'N/A'
            
        plt.text(bar.get_x() + bar.get_width()/2., height, label_text,
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path =  SAVE_NAME
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()