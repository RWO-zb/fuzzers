import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
import ast
import os
import csv

# ================= 配置区域 =================
SAVE_NAME = 'RQ3_Combined_Comparison_Seconds.png'
MAX_H = 12.0  # 实验时长 12小时

# 请确保以下文件路径正确
# 注意：这里根据你提供的代码片段推断了默认文件名
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
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 2
})

# ================= 数据处理函数 =================

def load_curefuzz(filepath):
    """逻辑来自 curefuzz-RQ3.py"""
    if not os.path.exists(filepath): return 0, np.nan
    try:
        with open(filepath, 'rb') as f: data = pickle.load(f)
        crashes = [entry for entry in data if entry.get('did_crash', False)]
        crashes.sort(key=lambda x: x.get('crash_time', 0) if x.get('crash_time') is not None else float('inf'))
        unique = []
        seen = set()
        for c in crashes:
            t = c.get('crash_time')
            if t is not None and t > MAX_H * 3600: continue
            state = c.get('mutate_state')
            if state is None: continue
            state_bytes = state.tobytes()
            if state_bytes not in seen:
                seen.add(state_bytes)
                unique.append(c)
        
        n = len(unique)
        if n == 0: return 0, np.nan
        
        # 指标计算 (改为秒)
        time_eff = (MAX_H * 3600) / n 
        gens = []
        for c in unique:
            p_depth = c.get('parent_depth')
            gens.append((p_depth + 1) if p_depth is not None else 1)
        return time_eff, np.mean(gens)
    except Exception as e:
        print(f"Error loading CureFuzz: {e}")
        return 0, np.nan

def load_gmodel(filepath):
    """逻辑来自 gmodel-RQ3.py"""
    if not os.path.exists(filepath): return 0, np.nan
    try:
        with open(filepath, 'rb') as f: data = pickle.load(f)
        if not data: return 0, np.nan
        
        crashes = [entry for entry in data if entry.get('is_crash', False)]
        unique = []
        seen = set()
        for c in crashes:
            inp = c.get('input')
            if inp is None: continue
            inp_tuple = tuple(inp) if isinstance(inp, list) else tuple(inp.tolist())
            if inp_tuple not in seen:
                seen.add(inp_tuple)
                unique.append(c)
        
        n = len(unique)
        if n == 0: return 0, np.nan
        
        # 指标计算 (改为秒)
        time_eff = (MAX_H * 3600) / n 
        return time_eff, np.nan 
    except Exception as e:
        print(f"Error loading G-Model: {e}")
        return 0, np.nan

def load_mdpfuzz_or_random(filepath, is_random):
    """逻辑来自 mdpfuzz-RQ3.py"""
    if not os.path.exists(filepath): return 0, np.nan
    try:
        unique = []
        seen = set()
        fuzz_start_time = None
        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            headers = next(reader, None)
            if not headers: return 0, np.nan
            headers = [h.strip() for h in headers]
            try:
                idx_input = headers.index('Input')
                idx_oracle = headers.index('Oracle')
                idx_gen = headers.index('Generation')
                idx_runtime = headers.index('RunTime')
            except ValueError: return 0, np.nan 
            
            rows = list(reader)
            rows.sort(key=lambda x: float(x[idx_runtime]) if x[idx_runtime].strip() != 'None' else 0)
            
            for row in rows:
                if not row: continue
                try:
                    run_time = float(row[idx_runtime])
                except: continue
                
                if fuzz_start_time is None: fuzz_start_time = run_time
                relative_time = run_time - fuzz_start_time
                
                if relative_time > MAX_H * 3600: continue
                
                gen_val = int(float(row[idx_gen]))
                oracle_str = row[idx_oracle].strip()
                inp_str = row[idx_input].strip()
                
                if oracle_str == 'True':
                    if inp_str not in seen:
                        seen.add(inp_str)
                        unique.append(gen_val)
        
        n = len(unique)
        if n == 0: return 0, np.nan if not is_random else 0, np.nan
        
        # 指标计算 (改为秒)
        time_eff = (MAX_H * 3600) / n 
        
        if is_random:
            return time_eff, np.nan
        else:
            return time_eff, np.mean(unique)
            
    except Exception as e:
        print(f"Error loading MDP/Random ({filepath}): {e}")
        return 0, np.nan

def load_qdfuzz(filepath):
    """逻辑来自 qdfuzz-RQ3.py"""
    if not os.path.exists(filepath): return 0, np.nan
    try:
        df = pd.read_csv(filepath)
        crashes = []
        for _, row in df.iterrows():
            if int(row['mutation_count']) == 0: continue
            if not bool(row['is_faulty']): continue
            if row['discovery_time'] > MAX_H * 3600: continue
            
            inp = row['input']
            if isinstance(inp, str):
                try: inp_val = np.array(json.loads(inp))
                except: 
                    try: inp_val = np.array(ast.literal_eval(inp))
                    except: inp_val = inp
            else: inp_val = inp
            
            crashes.append({
                'state': inp_val,
                'gen': int(row['mutation_count']) + 1
            })
            
        unique = []
        seen = set()
        for c in crashes:
            state = c['state']
            if hasattr(state, 'tobytes'): key = state.tobytes()
            else: key = str(state)
            if key not in seen:
                seen.add(key)
                unique.append(c['gen'])
                
        n = len(unique)
        if n == 0: return 0, np.nan
        
        # 指标计算 (改为秒)
        time_eff = (MAX_H * 3600) / n 
        return time_eff, np.mean(unique)
    except Exception as e:
        print(f"Error loading QDFuzz: {e}")
        return 0, np.nan

def load_seqfuzz(filepath):
    """逻辑来自 seqfuzz-RQ3.py"""
    if not os.path.exists(filepath): return 0, np.nan
    try:
        with open(filepath, 'rb') as f: data = pickle.load(f)
        crashes = [entry for entry in data if entry.get('crashed', False)]
        crashes.sort(key=lambda x: x.get('crash_time', 0) if x.get('crash_time') is not None else float('inf'))
        
        unique_gens = []
        seen = set()
        for c in crashes:
            t = c.get('crash_time')
            if t is not None and t > MAX_H * 3600: continue
            state = c.get('state')
            if state is None: continue
            try: state_bytes = state.tobytes() if hasattr(state, 'tobytes') else bytes(state)
            except: continue
            
            if state_bytes not in seen:
                seen.add(state_bytes)
                gen = c.get('generation')
                if gen is not None: unique_gens.append(gen)
                else: unique_gens.append(0) 
                
        n = len(unique_gens)
        if n == 0: return 0, np.nan
        
        # 指标计算 (改为秒)
        time_eff = (MAX_H * 3600) / n 
        return time_eff, np.mean(unique_gens)
    except Exception as e:
        print(f"Error loading SeqFuzz: {e}")
        return 0, np.nan

# ================= 主程序 =================

def main():
    # 定义处理顺序 (字母顺序，确保颜色映射一致)
    methods_order = ["CureFuzz", "G-Model", "MDPFuzz", "QDFuzz", "Random", "SeqFuzz"]
    
    metrics = {
        "labels": [],
        "time_per_crash": [],
        "gen_avg_depth": []
    }
    
    print("Starting data processing...")
    
    for label in methods_order:
        path = FILE_PATHS.get(label)
        t_eff, g_avg = 0, np.nan
        
        print(f"Processing {label}...", end=" ")
        
        if label == "CureFuzz":
            t_eff, g_avg = load_curefuzz(path)
        elif label == "G-Model":
            t_eff, g_avg = load_gmodel(path)
        elif label == "MDPFuzz":
            t_eff, g_avg = load_mdpfuzz_or_random(path, is_random=False)
        elif label == "QDFuzz":
            t_eff, g_avg = load_qdfuzz(path)
        elif label == "Random":
            t_eff, g_avg = load_mdpfuzz_or_random(path, is_random=True)
        elif label == "SeqFuzz":
            t_eff, g_avg = load_seqfuzz(path)
            
        print(f"TimeEff: {t_eff:.2f} s, Gen: {g_avg}")
        
        metrics["labels"].append(label)
        metrics["time_per_crash"].append(t_eff)
        metrics["gen_avg_depth"].append(g_avg)

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.arange(len(metrics["labels"])))

    # Subplot 1: Time Efficiency
    ax1 = axes[0]
    bars1 = ax1.bar(metrics["labels"], metrics["time_per_crash"], 
                    color=colors, alpha=0.8, edgecolor='black', width=0.6)
    ax1.set_title("Time Cost Efficiency")
    ax1.set_ylabel("Avg. Seconds per Crash") # 修改 Y 轴标签
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars1:
        height = bar.get_height()
        label_text = f'{height:.1f} s' if height > 0 else 'N/A' # 修改单位为 s
        ax1.text(bar.get_x() + bar.get_width()/2., height, label_text,
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Subplot 2: Generation Efficiency
    ax2 = axes[1]
    
    # 筛选有效数据用于绘图 (去掉 NaN)
    valid_indices = [i for i, x in enumerate(metrics["gen_avg_depth"]) if not np.isnan(x)]
    valid_labels = [metrics["labels"][i] for i in valid_indices]
    valid_values = [metrics["gen_avg_depth"][i] for i in valid_indices]
    valid_colors = [colors[i] for i in valid_indices]

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
    ax2.set_ylabel("Avg. Generation Index")
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(SAVE_NAME, dpi=300)
    print(f"\nPlot saved to {SAVE_NAME}")
    plt.show()

if __name__ == "__main__":
    main()