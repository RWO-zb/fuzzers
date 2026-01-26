import matplotlib
# 强制使用非交互式后端，防止 IDE/Server 环境下绘图冲突导致保存空白
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pickle
import json
import ast
import os
import csv

# ================= 配置区域 =================
# 建议保持 PDF 格式，我们会通过代码修复渲染问题
SAVE_BAR_NAME = 'RQ3_Combined_Comparison_Seconds.pdf'
SAVE_BOX_NAME = 'RQ3_Generation_Boxplot.pdf'
SAVE_BOX_NAME_PNG = 'RQ3_Generation_Boxplot_Preview.png' # 同时保存PNG方便快速查看

MAX_H = 12.0  # 实验时长 12小时

# G-Model 周期配置
G_MODEL_ROUND_LEN = 100 
G_MODEL_RANDOM_LEN = 50 

# 请确保以下文件路径正确
FILE_PATHS = {
    "CureFuzz": "selection_log.pkl",
    "G-Model":  "all_test_cases_log.pkl",
    "MDPFuzz":  "MC_DQN_NoCov_5_0.01_0.1_0_12h_logs.txt", 
    "QDFuzz":   "mc_test_data.csv",
    "Random":   "MC_DQN_RT_0_budget730000_logs.txt",     
    "SeqFuzz":  "all_run_seeds_0.pkl"
}

# ================= 绘图样式设置 =================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    # 保证 PDF 字体嵌入，方便后期编辑
    'pdf.fonttype': 42, 
    'ps.fonttype': 42
})

# ================= 数据处理函数 (逻辑保持不变) =================

def load_curefuzz(filepath):
    if not os.path.exists(filepath): 
        print(f"[Warn] File not found: {filepath}")
        return 0, np.nan, []
    try:
        with open(filepath, 'rb') as f: data = pickle.load(f)
        crashes = [entry for entry in data if entry.get('did_crash', False)]
        crashes.sort(key=lambda x: x.get('elapsed_time', 0))
        unique_gens = []
        seen = set()
        for c in crashes:
            t = c.get('elapsed_time')
            if t is not None and t > MAX_H * 3600: continue
            state = c.get('mutate_state')
            if state is None: continue
            state_bytes = state.tobytes() if hasattr(state, 'tobytes') else str(state)
            
            if state_bytes not in seen:
                seen.add(state_bytes)
                unique_gens.append(c.get('parent_depth', 0) + 1)
        
        n = len(unique_gens)
        if n == 0: return 0, np.nan, []
        time_eff = (MAX_H * 3600) / n 
        return time_eff, np.mean(unique_gens), unique_gens
    except Exception as e:
        print(f"Error loading CureFuzz: {e}")
        return 0, np.nan, []

def load_gmodel(filepath):
    if not os.path.exists(filepath): 
        print(f"[Warn] File not found: {filepath}")
        return 0, np.nan, []
    try:
        with open(filepath, 'rb') as f: data = pickle.load(f)
        if not data: return 0, np.nan, []
        
        crashes = [entry for entry in data if entry.get('is_crash', False)]
        crashes.sort(key=lambda x: x.get('time', 0))
        unique_gens = []
        seen = set()
        for c in crashes:
            t = c.get('time')
            if t is not None and t > MAX_H * 3600: continue
            step = c.get('step', 0)
            phase_offset = step % G_MODEL_ROUND_LEN
            if phase_offset < G_MODEL_RANDOM_LEN: continue
            current_gen = int(step / G_MODEL_ROUND_LEN) + 1
            inp = c.get('input')
            if inp is None: continue
            inp_tuple = tuple(inp) if isinstance(inp, list) else tuple(inp.tolist())
            if inp_tuple not in seen:
                seen.add(inp_tuple)
                unique_gens.append(current_gen)
        n = len(unique_gens)
        if n == 0: return 0, np.nan, []
        time_eff = (MAX_H * 3600) / n 
        return time_eff, np.mean(unique_gens), unique_gens
    except Exception as e:
        print(f"Error loading G-Model: {e}")
        return 0, np.nan, []

def load_mdpfuzz_or_random(filepath, is_random):
    if not os.path.exists(filepath): 
        print(f"[Warn] File not found: {filepath}")
        return 0, np.nan, []
    try:
        unique_gens = []
        seen = set()
        fuzz_start_time = None
        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            headers = next(reader, None)
            if not headers: return 0, np.nan, []
            headers = [h.strip() for h in headers]
            try:
                idx_input = headers.index('Input')
                idx_oracle = headers.index('Oracle')
                idx_gen = headers.index('Generation')
                idx_runtime = headers.index('RunTime')
            except ValueError: return 0, np.nan, []
            idx_crashtime = headers.index('CrashTime') if 'CrashTime' in headers else -1
            rows = list(reader)
            rows.sort(key=lambda x: float(x[idx_runtime]) if x[idx_runtime].strip() != 'None' else 0)
            for row in rows:
                if not row: continue
                try: gen_val = int(float(row[idx_gen]))
                except: continue
                if not is_random and gen_val == 0: continue 
                if row[idx_oracle].strip() != 'True': continue
                relative_time = None
                if idx_crashtime != -1:
                    ct_str = row[idx_crashtime].strip()
                    if ct_str != 'None': relative_time = float(ct_str)
                if relative_time is None:
                    try:
                        run_time = float(row[idx_runtime])
                        if fuzz_start_time is None: fuzz_start_time = run_time
                        relative_time = run_time - fuzz_start_time
                    except: continue
                if relative_time > MAX_H * 3600: continue
                inp_str = row[idx_input].strip()
                if inp_str not in seen:
                    seen.add(inp_str)
                    if not is_random: unique_gens.append(gen_val)
        n = len(seen)
        if n == 0: return 0, np.nan, []
        time_eff = (MAX_H * 3600) / n 
        if is_random: return time_eff, np.nan, []
        else: return time_eff, np.mean(unique_gens), unique_gens
    except Exception as e:
        print(f"Error loading MDP/Random: {e}")
        return 0, np.nan, []

def load_qdfuzz(filepath):
    if not os.path.exists(filepath): 
        print(f"[Warn] File not found: {filepath}")
        return 0, np.nan, []
    try:
        df = pd.read_csv(filepath)
        unique_gens = []
        seen = set()
        for _, row in df.iterrows():
            if int(row['mutation_count']) == 0: continue
            if not bool(row['is_faulty']): continue
            if row['discovery_time'] > MAX_H * 3600: continue
            inp = row['input']
            if isinstance(inp, str):
                try: inp_val = np.array(json.loads(inp))
                except: inp_val = inp 
            else: inp_val = inp
            key = inp_val.tobytes() if hasattr(inp_val, 'tobytes') else str(inp_val)
            if key not in seen:
                seen.add(key)
                unique_gens.append(int(row['mutation_count']) + 1)
        n = len(unique_gens)
        if n == 0: return 0, np.nan, []
        time_eff = (MAX_H * 3600) / n 
        return time_eff, np.mean(unique_gens), unique_gens
    except Exception as e:
        print(f"Error loading QDFuzz: {e}")
        return 0, np.nan, []

def load_seqfuzz(filepath):
    if not os.path.exists(filepath): 
        print(f"[Warn] File not found: {filepath}")
        return 0, np.nan, []
    try:
        with open(filepath, 'rb') as f: data = pickle.load(f)
        crashes = [entry for entry in data if entry.get('crashed', False)]
        crashes.sort(key=lambda x: x.get('crash_time', 0) if x.get('crash_time') is not None else float('inf'))
        unique_gens = []
        seen = set()
        for c in crashes:
            t = c.get('crash_time')
            if t is not None and t > MAX_H * 3600: continue
            if c.get('generation', 0) == 0: continue
            state = c.get('state')
            if state is None: continue
            state_bytes = state.tobytes() if hasattr(state, 'tobytes') else bytes(state)
            if state_bytes not in seen:
                seen.add(state_bytes)
                unique_gens.append(c.get('generation'))
        n = len(unique_gens)
        if n == 0: return 0, np.nan, []
        time_eff = (MAX_H * 3600) / n 
        return time_eff, np.mean(unique_gens), unique_gens
    except Exception as e:
        print(f"Error loading SeqFuzz: {e}")
        return 0, np.nan, []

# ================= [核心美化] 绘图函数 =================

def plot_generation_distribution(gen_data_map, color_map, save_path):
    # 1. 验证数据
    valid_data = {k: v for k, v in gen_data_map.items() if v and len(v) > 0}
    if not valid_data:
        print("[Error] No valid data found for boxplot!")
        return

    # 打印数据摘要
    print("\n--- Boxplot Data Summary ---")
    min_val, max_val = float('inf'), float('-inf')
    for k, v in valid_data.items():
        v_min, v_max = min(v), max(v)
        min_val = min(min_val, v_min)
        max_val = max(max_val, v_max)
        print(f"  {k:<10}: Count={len(v)}, Range=[{v_min}, {v_max}]")
    print("----------------------------\n")

    labels = list(valid_data.keys())
    data_values = list(valid_data.values())
    colors = [mcolors.to_hex(color_map.get(lbl, '#333333')) for lbl in labels]

    # 创建 Figure
    fig, ax = plt.subplots(figsize=(12, 7)) # 稍微加大画布

    # 2. 绘制散点层 (底坑) -- Raincloud 风格
    # 关键修改：先画散点，且 zorder=1，让它位于最底层
    for i, (method, values) in enumerate(valid_data.items()):
        y_pos = i + 1
        # 添加垂直抖动
        y_jitter = np.random.normal(y_pos, 0.08, size=len(values))
        
        # [美化] 
        # alpha=0.3: 提高透明度，密集时变深，稀疏时可见
        # edgecolor='none': 去掉白边，避免点太密时全是一片白
        # zorder=1: 放在箱线图下面
        # rasterized=True: 确保 PDF 不会过大且不空白
        ax.scatter(values, y_jitter, alpha=0.3, 
                   color=colors[i], 
                   s=12, marker='o', edgecolor='none', 
                   zorder=1, rasterized=True)

    # 3. 绘制箱线图层 (顶层)
    # [美化] 
    # zorder=10: 放在散点上面
    # showfliers=False: 不显示离群点（因为散点已经画了所有点，没必要重复）
    # widths=0.5: 稍微调窄一点，让散点露出来更多
    box = ax.boxplot(data_values, vert=False, patch_artist=True,
                     labels=labels, showmeans=True,
                     widths=0.5, showfliers=False, zorder=10,
                     meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":7, "zorder":11},
                     medianprops={"color": "black", "linewidth": 1.5, "zorder":11},
                     boxprops={"linewidth": 1.2, "zorder":10},
                     whiskerprops={"linewidth": 1.2, "zorder":10},
                     capprops={"linewidth": 1.2, "zorder":10})

    # 给箱体上色
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8) # 提高不透明度，遮住后面的散点，突出箱体
        # 强制箱体本身栅格化，防止部分 PDF 阅读器渲染透明度 bug
        patch.set_rasterized(True) 

    # 4. 坐标轴与装饰
    ax.set_title('Distribution of Unique Crashes by Generation (Generative Phase Only)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Generation / Round Index (Log Scale)', fontsize=14)
    
    # 使用 Log 坐标轴
    ax.set_xscale('log')
    
    # 设置 X 轴显示范围 (留出余量)
    if min_val > 0:
        ax.set_xlim(left=max(0.8, min_val * 0.7), right=max_val * 2.0)
    
    # 格式化刻度
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, which="both", ls="--", alpha=0.3, zorder=0) # 网格放在最底层

    plt.tight_layout()
    
    # 保存文件
    try:
        # 保存 PDF (论文用)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Success] Boxplot PDF saved to {save_path}")
        
        # 保存 PNG (预览用)
        if SAVE_BOX_NAME_PNG:
            fig.savefig(SAVE_BOX_NAME_PNG, dpi=300, bbox_inches='tight')
            print(f"[Success] Boxplot PNG saved to {SAVE_BOX_NAME_PNG}")
            
    except Exception as e:
        print(f"[Error] Failed to save boxplot: {e}")
    finally:
        plt.close(fig)

# ================= 主程序 =================

def main():
    methods_order = ["CureFuzz", "G-Model", "MDPFuzz", "QDFuzz", "Random", "SeqFuzz"]
    
    metrics = {
        "labels": [],
        "time_per_crash": [],
        "gen_avg_depth": []
    }
    boxplot_data = {}
    
    print(f"Starting processing (Max Hours: {MAX_H})...")
    
    for label in methods_order:
        path = FILE_PATHS.get(label)
        t_eff, g_avg, g_list = 0, np.nan, []
        
        if label == "CureFuzz":
            t_eff, g_avg, g_list = load_curefuzz(path)
        elif label == "G-Model":
            t_eff, g_avg, g_list = load_gmodel(path)
        elif label == "MDPFuzz":
            t_eff, g_avg, g_list = load_mdpfuzz_or_random(path, is_random=False)
        elif label == "QDFuzz":
            t_eff, g_avg, g_list = load_qdfuzz(path)
        elif label == "Random":
            t_eff, g_avg, g_list = load_mdpfuzz_or_random(path, is_random=True)
        elif label == "SeqFuzz":
            t_eff, g_avg, g_list = load_seqfuzz(path)
            
        print(f"{label}: Time={t_eff:.1f}s, AvgGen={g_avg:.1f}, Samples={len(g_list)}")
        
        metrics["labels"].append(label)
        metrics["time_per_crash"].append(t_eff)
        metrics["gen_avg_depth"].append(g_avg)
        
        if len(g_list) > 0 and label != "Random":
            boxplot_data[label] = g_list

    # --- 颜色准备 ---
    tab10 = plt.cm.tab10(np.arange(len(methods_order)))
    color_map = {label: color for label, color in zip(methods_order, tab10)}

    # --- 绘图 1: 柱状图 ---
    print("\nGenerating Bar Charts...")
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar 1
    ax1 = axes[0]
    bars1 = ax1.bar(metrics["labels"], metrics["time_per_crash"], 
                    color=tab10, alpha=0.8, edgecolor='black', width=0.6)
    ax1.set_title("Time Cost Efficiency (Generative Phase Only)")
    ax1.set_ylabel("Avg. Seconds per Crash") 
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars1:
        height = bar.get_height()
        label_text = f'{height:.1f} s' if height > 0 else 'N/A'
        ax1.text(bar.get_x() + bar.get_width()/2., height, label_text,
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Bar 2
    ax2 = axes[1]
    valid_indices = [i for i, x in enumerate(metrics["gen_avg_depth"]) if not np.isnan(x)]
    valid_labels = [metrics["labels"][i] for i in valid_indices]
    valid_values = [metrics["gen_avg_depth"][i] for i in valid_indices]
    valid_colors = [tab10[i] for i in valid_indices]

    if valid_indices:
        bars2 = ax2.bar(valid_labels, valid_values, 
                        color=valid_colors, alpha=0.8, edgecolor='black', width=0.6)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Generation Data', ha='center', va='center', transform=ax2.transAxes)

    ax2.set_title("Average Discovery Generation")
    ax2.set_ylabel("Avg. Generation / Round Index")
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    fig1.savefig(SAVE_BAR_NAME, dpi=300, bbox_inches='tight')
    plt.close(fig1) 
    print(f"Bar charts saved to {SAVE_BAR_NAME}")

    # --- 绘图 2: 箱线图 (美化版) ---
    print("\nGenerating Beautified Boxplot...")
    plot_generation_distribution(boxplot_data, color_map, SAVE_BOX_NAME)
    
    print("Done.")

if __name__ == "__main__":
    main()