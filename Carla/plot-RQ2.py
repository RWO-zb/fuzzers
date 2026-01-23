import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================

# 1. 核心设置
MAX_SEEDS = 1000         # 限制只取前1000个种子
OUTPUT_FILE = 'RQ2-CARLA_Final_Counts.png'

# [关键修改] 根据代码分析，State Coverage 使用 100x100 网格
# 因此总格子数 (分母) 为 10,000
TOTAL_GRID_SIZE = 10000  

# 2. 绘图样式
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

# 3. 文件与图例配置
# 请确保 CSV 文件名与实际一致
files_config = {
    "curefuzz.csv": {
        "label": "CureFuzz", 
        "phase_col": "phase", 
        "target_phase": "Phase2",
        "color": "#1f77b4"  # C0 Blue
    },
    "g-model.csv": {
        "label": "G-Model", 
        "special": "g-model",
        "color": "#ff7f0e"  # C1 Orange
    },
    "mdpfuzz.csv": {
        "label": "MDPFuzz", 
        "phase_col": "phase", 
        "target_phase": "Phase2", # MDPFuzz 通常也有搜索阶段，这里假设筛选 Phase2 或对应的主阶段
        "color": "#2ca02c"  # C2 Green
    },
    "qdfuzz.csv": {
        "label": "QDFuzz", 
        "phase_col": "phase", 
        "target_phase": "Phase2",
        "color": "#d62728"  # C3 Red
    },
    "random.csv": {
        "label": "Random", 
        "special": "random",
        "color": "#9467bd"  # C4 Purple
    },
    "seqfuzz.csv": {
        "label": "SeqFuzz", 
        "phase_col": "phase", 
        "target_phase": "Phase2",
        "color": "#8c564b"  # C5 Brown
    }
}

# 4. 指标配置
metrics = [
    # Y轴标签明确为 "计数" (#)
    {"col": "state_coverage", "title": "State Coverage Growth", "ylabel": "# Unique States"},
    {"col": "behavior_count", "title": "Behavior Diversity Growth", "ylabel": "# Unique Behaviors"},
    {"col": "fault_behavior_count", "title": "Fault Diversity Growth", "ylabel": "# Unique Faults"}
]

# ================= 辅助函数 =================

def print_final_stats(final_stats):
    """打印最终统计数据的表格"""
    print("\n" + "="*75)
    print(f"{'Method':<15} | {'Unique States':<15} | {'Behav Div':<12} | {'Fault Div':<12}")
    print("-" * 75)
    
    sorted_names = sorted(final_stats.keys())
    
    for name in sorted_names:
        stats = final_stats[name]
        sc = stats.get('state_coverage', 0)
        bd = stats.get('behavior_count', 0)
        fd = stats.get('fault_behavior_count', 0)
        
        # 打印为整数
        print(f"{name:<15} | {int(sc):<15} | {int(bd):<12} | {int(fd):<12}")
    
    print("="*75 + "\n")

# ================= 主逻辑 =================

def main():
    data_store = {m["col"]: {} for m in metrics}
    final_stats = {}

    print("--- Loading Data ---")

    for fname, config in files_config.items():
        label = config["label"]
        if label not in final_stats:
            final_stats[label] = {}
            
        try:
            # 读取 CSV
            try:
                df = pd.read_csv(fname)
            except FileNotFoundError:
                print(f"[Warning] File not found: {fname}, skipping.")
                continue
            
            # ---------------- 核心筛选逻辑 ----------------
            df_filtered = pd.DataFrame()

            # 1. G-Model 特殊处理：寻找 generative+novelty 阶段起点
            if "special" in config and config["special"] == "g-model":
                if 'method' in df.columns and 'generative+novelty' in df['method'].values:
                    start_idx = df[df['method'] == 'generative+novelty'].index[0]
                    df_filtered = df.loc[start_idx:].copy()
                else:
                    df_filtered = df.copy()

            # 2. Random：直接取全部
            elif "special" in config and config["special"] == "random":
                df_filtered = df.copy()
                
            # 3. 其他 Fuzzers：筛选 Phase2
            else:
                target_phase = config["target_phase"]
                phase_col = config.get("phase_col", "phase")
                
                # 检查列是否存在
                if phase_col in df.columns:
                    # 如果该阶段存在，则筛选
                    if target_phase in df[phase_col].values:
                        df_filtered = df[df[phase_col] == target_phase].copy()
                    else:
                        # 如果没找到 Phase2，有些 csv 可能没有 phase 列或者叫法不同，这里做个回退
                        print(f"[Info] {label}: '{target_phase}' not found, using all data.")
                        df_filtered = df.copy()
                else:
                    df_filtered = df.copy()

            # ---------------- 数量截取 ----------------
            # 只取前 MAX_SEEDS 个
            if len(df_filtered) > MAX_SEEDS:
                df_filtered = df_filtered.iloc[:MAX_SEEDS]
            
            # 重置索引，确保绘图从 1 开始
            df_filtered = df_filtered.reset_index(drop=True)
            
            # ---------------- 提取指标并转换 ----------------
            for m in metrics:
                col_name = m["col"]
                if col_name in df_filtered.columns:
                    values = df_filtered[col_name].values
                    
                    # [关键逻辑] 处理 State Coverage 归一化问题
                    if col_name == "state_coverage":
                        # 如果数据是归一化的 (最大值 <= 1.0)，则乘以 TOTAL_GRID_SIZE
                        # 注意：如果某次实验只覆盖了极少状态(例如0.0001)，它也是<=1.0，这逻辑兼容
                        # 但如果你的CSV里已经是计数(例如 150)，则不会触发此逻辑(假设150 > 1.0)
                        if len(values) > 0 and np.max(values) <= 1.0:
                            values = values * TOTAL_GRID_SIZE
                    
                    data_store[col_name][label] = values
                    
                    # 记录最终值用于表格
                    if len(values) > 0:
                        final_stats[label][col_name] = values[-1]
                    else:
                        final_stats[label][col_name] = 0
                else:
                    pass # 某些文件可能缺失特定列

            print(f"Loaded {label}: {len(df_filtered)} seeds.")

        except Exception as e:
            print(f"[Error] Processing {fname}: {e}")

    # --- 打印统计表格 ---
    print_final_stats(final_stats)

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, m in enumerate(metrics):
        ax = axes[i]
        col_name = m["col"]
        
        # 按配置顺序绘图
        for fname, config in files_config.items():
            label = config["label"]
            if label in data_store[col_name]:
                values = data_store[col_name][label]
                
                # X轴：测试用例数量 (1 到 N)
                x_axis = range(1, len(values) + 1)
                
                ax.plot(x_axis, values, label=label, color=config.get("color"), alpha=0.9, linewidth=2)
                
        ax.set_title(m["title"], fontweight='bold', fontsize=14)
        ax.set_xlabel("Number of Test Cases", fontsize=12)
        ax.set_ylabel(m["ylabel"], fontsize=12)
        
        # 设置范围
        ax.set_xlim(left=0, right=MAX_SEEDS)
        ax.set_ylim(bottom=0) 
        
        # 网格线
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 仅第一张图显示图例
        if i == 0:
            ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Plot saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()