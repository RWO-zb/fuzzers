import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 全局配置 ---
# 设置学术风格
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

# 文件配置 (与之前保持一致，用于对齐时间)
files_config = {
    "curefuzz.csv": {
        "label": "CureFuzz", 
        "time_col": "elapsed_time", 
        "phase_col": "phase", 
        "target_phase": "Phase2"
    },
    "g-model.csv": {
        "label": "G-Model", 
        "time_col": "elapsed_time", 
        "special": "g-model"
    },
    "mdpfuzz.csv": {
        "label": "MDPFuzz", 
        "time_col": "global_time", 
        "phase_col": "phase", 
        "target_phase": "Phase2"
    },
    "qdfuzz.csv": {
        "label": "QDFuzz", 
        "time_col": "elapsed_time", 
        "phase_col": "phase", 
        "target_phase": "Phase2"
    },
    "random.csv": {
        "label": "Random", 
        "time_col": "global_time", 
        "special": "random"
    },
    "seqfuzz.csv": {
        "label": "SeqFuzz", 
        "time_col": "elapsed_time", 
        "phase_col": "phase", 
        "target_phase": "Phase2"
    }
}

# 绘图参数
max_h = 12.0
metrics = [
    {"col": "state_coverage", "title": "Cumulative State Coverage", "ylabel": "State Coverage"},
    {"col": "behavior_count", "title": "Cumulative Behavior Diversity", "ylabel": "# Unique Behaviors"},
    {"col": "fault_behavior_count", "title": "Cumulative Fault Diversity", "ylabel": "# Unique Faults"}
]

# --- 2. 数据读取与预处理 ---
# 存储结构: data_store[metric][fname] = (times, values)
data_store = {m["col"]: {} for m in metrics}

for fname, config in files_config.items():
    try:
        df = pd.read_csv(fname)
        
        # --- 时间筛选逻辑 (复用之前的逻辑) ---
        if "special" in config and config["special"] == "g-model":
            if 'generative+novelty' in df['method'].values:
                start_time = df[df['method'] == 'generative+novelty'][config['time_col']].min()
            else:
                start_time = df[config['time_col']].min()
            df_filtered = df[df[config['time_col']] >= start_time].copy()
            
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
                print(f"Warning: {target_phase} not found in {fname}")
                continue
                
        # 时间归一化
        df_filtered['norm_time'] = df_filtered[config['time_col']] - start_time
        
        # 截取前12小时
        limit_sec = max_h * 3600
        df_filtered = df_filtered[df_filtered['norm_time'] <= limit_sec].copy()
        
        # 排序
        df_filtered = df_filtered.sort_values('norm_time')
        
        # --- 提取各指标数据 ---
        for m in metrics:
            col_name = m["col"]
            if col_name in df_filtered.columns:
                # 提取 (时间, 值) 对
                # 假设CSV里已经是累积值，直接取值即可
                times = df_filtered['norm_time'].values
                values = df_filtered[col_name].values
                data_store[col_name][fname] = (times, values)
            else:
                print(f"Warning: Metric {col_name} not found in {fname}")
                
    except Exception as e:
        print(f"Error processing {fname}: {e}")

# --- 3. 绘图 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, m in enumerate(metrics):
    ax = axes[i]
    col_name = m["col"]
    
    for fname, config in files_config.items():
        if fname in data_store[col_name]:
            times, values = data_store[col_name][fname]
            
            # 转换为小时
            times_h = times / 3600.0
            
            # 绘制曲线
            # 使用 step 或 plot 均可。对于累积增长，plot 看起来更平滑，step 更精确。
            # 这里为了看清增长趋势，使用 plot
            # 添加起点 (0,0) 以便从原点开始
            if len(times_h) > 0:
                # 只有当第一个点不是0时刻才添加原点 (可选，视数据情况而定)
                # 这里简单处理：直接绘制提取的数据
                ax.plot(times_h, values, label=config["label"])
            
    ax.set_title(m["title"])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(m["ylabel"])
    ax.set_xlim(0, max_h)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 只在第一个图显示图例，避免重复
    if i == 0:
        ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('RQ2-CARLA.png', dpi=300)
plt.show()