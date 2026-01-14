import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 全局配置 ---
# 设置学术风格
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

# 文件配置
files_config = {
    "curefuzz.csv": {
        "label": "CureFuzz", 
        "time_col": "elapsed_time", 
        "gen_col": "mutation_generation",
        "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"
    },
    "g-model.csv": {
        "label": "G-Model", 
        "time_col": "elapsed_time", 
        "gen_col": None, # G-Model 无变异代数
        "special": "g-model", "input_col": "input_post"
    },
    "mdpfuzz.csv": {
        "label": "MDPFuzz", 
        "time_col": "global_time", 
        "gen_col": "generation",
        "phase_col": "phase", "target_phase": "Phase2", "input_col": "current_input"
    },
    "qdfuzz.csv": {
        "label": "QDFuzz", 
        "time_col": "elapsed_time", 
        "gen_col": "mutation_generation",
        "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"
    },
    "random.csv": {
        "label": "Random", 
        "time_col": "global_time", 
        "gen_col": None, # Random 无变异代数
        "special": "random", "input_col": "current_input"
    },
    "seqfuzz.csv": {
        "label": "SeqFuzz", 
        "time_col": "elapsed_time", 
        "gen_col": "mutation_generation",
        "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"
    }
}

# 统计参数
max_h = 12.0 # 统计前12小时
metrics_data = {
    "labels": [],
    "time_per_crash": [], # Minutes per crash (Cost)
    "gen_avg_depth": []   # Avg. Generation of Discovery (Sum(gens)/N)
}

# --- 2. 数据处理与指标计算 ---
for fname, config in files_config.items():
    try:
        df = pd.read_csv(fname)
        label = config["label"]
        metrics_data["labels"].append(label)
        
        # --- 数据筛选 (复用之前的逻辑) ---
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
                metrics_data["time_per_crash"].append(0)
                metrics_data["gen_avg_depth"].append(np.nan)
                continue

        # 时间归一化
        df_filtered['norm_time'] = df_filtered[config['time_col']] - start_time
        
        # 截取前 12 小时的数据
        limit_sec = max_h * 3600
        df_12h = df_filtered[df_filtered['norm_time'] <= limit_sec].copy()
        
        # --- 1. 提取 Unique Crashes ---
        # 筛选 Crash
        if df_12h['success'].dtype == 'bool':
             is_crash = df_12h['success'] == False
        else:
             is_crash = df_12h['success'].astype(str) == 'False'
        
        crashes = df_12h[is_crash].copy()
        input_col = config.get("input_col")
        
        if not crashes.empty and input_col in crashes.columns:
            # 必须按照时间排序，确保保留的是第一次发现该Crash时的记录（包括当时的代数）
            crashes = crashes.sort_values('norm_time')
            unique_crashes = crashes.drop_duplicates(subset=[input_col], keep='first')
            n_crashes = len(unique_crashes)
        else:
            unique_crashes = pd.DataFrame()
            n_crashes = 0
            
        # --- 2. 计算 Time Efficiency (Minutes / Crash) ---
        # 定义：总测试时长 / 发现的Crash总数 (资源消耗视角)
        if n_crashes > 0:
            time_eff = (max_h * 60) / n_crashes 
        else:
            time_eff = 0 
            
        metrics_data["time_per_crash"].append(time_eff)
        
        # --- 3. 计算 Average Generation (Sum(Gens) / N) ---
        # 定义：所有Crash发现时的代数之和 / Crash总数 (按照你的最新要求)
        gen_col = config.get("gen_col")
        
        if gen_col and gen_col in unique_crashes.columns and n_crashes > 0:
            # 计算总和
            total_gens = unique_crashes[gen_col].sum()
            # 计算平均
            avg_gen = total_gens / n_crashes
            metrics_data["gen_avg_depth"].append(avg_gen)
        else:
            metrics_data["gen_avg_depth"].append(np.nan) # N/A for Random/G-Model or 0 crashes
            
    except Exception as e:
        print(f"Error processing {fname}: {e}")

# --- 3. 绘图 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = plt.cm.tab10(np.arange(len(metrics_data["labels"])))

# --- Subplot 1: Time Efficiency ---
ax1 = axes[0]
bars1 = ax1.bar(metrics_data["labels"], metrics_data["time_per_crash"], color=colors, alpha=0.8, edgecolor='black')
ax1.set_title("Time Cost Efficiency")
ax1.set_ylabel("Avg. Minutes per Crash")
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# 数值标签
for bar in bars1:
    height = bar.get_height()
    label_text = f'{height:.1f} min' if height > 0 else 'N/A'
    ax1.text(bar.get_x() + bar.get_width()/2., height, label_text,
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Subplot 2: Generation Efficiency ---
ax2 = axes[1]
# 过滤掉无效数据
valid_indices = [i for i, x in enumerate(metrics_data["gen_avg_depth"]) if not np.isnan(x)]
valid_labels = [metrics_data["labels"][i] for i in valid_indices]
valid_values = [metrics_data["gen_avg_depth"][i] for i in valid_indices]
valid_colors = [colors[i] for i in valid_indices]

bars2 = ax2.bar(valid_labels, valid_values, color=valid_colors, alpha=0.8, edgecolor='black')
ax2.set_title("Average Discovery Generation")
ax2.set_ylabel("Avg. Generation Index")
ax2.grid(axis='y', linestyle='--', alpha=0.6)

# 数值标签
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('RQ3-CARLA.png', dpi=300)
plt.show()