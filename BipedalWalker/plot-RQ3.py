import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json

# ==========================================
# 1. 全局配置与风格设置
# ==========================================
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

# --- 颜色映射修改 ---
# 为了与 plot-RQ3.py 保持一致，这里使用 Tab10 调色板的对应颜色
# 对应顺序: CureFuzz(0), G-Model(1), MDPFuzz(2), QDFuzz(3), Random(4), SeqFuzz(5)
COLOR_MAP = {
    "CureFuzz": "#1f77b4", # Tab10[0] Blue
    "G-Model":  "#ff7f0e", # Tab10[1] Orange
    "MDPFuzz":  "#2ca02c", # Tab10[2] Green
    "QDFuzz":   "#d62728", # Tab10[3] Red
    "Random":   "#9467bd", # Tab10[4] Purple
    "SeqFuzz":  "#8c564b", # Tab10[5] Brown
}

# 统计截断时间 (小时)
MAX_HOURS = 12.0

# ==========================================
# 2. 文件路径配置
# ==========================================
# --- 顺序修改 ---
# 字典顺序调整为字母顺序，以确保柱状图从左到右的排列与 plot-RQ3.py 一致
FILES_CONFIG = {
    "CureFuzz": {
        "path": "selection_log.pkl",               
        "type": "cure_pkl"
    },
    "G-Model": {
        "path": "all_test_cases_log.pkl",          
        "type": "gmodel_pkl"
    },
    "MDPFuzz": {
        "path": "fuzzer_10_0.01_0.01_0_logs.txt", 
        "type": "mdpfuzz_txt" 
    },
    "QDFuzz": {
        "path": "1765639810.5339673_data.csv",                 
        "type": "qdfuzz_csv"
    },
    "Random": {
        "path": "rt_10_0.01_0.01_0_logs.txt",          
        "type": "mdpfuzz_txt"
    },
    "SeqFuzz": {
        "path": "all_run_seeds_0.pkl",             
        "type": "seq_pkl"
    }
}

# ==========================================
# 3. 数据加载器 (保持原有逻辑不变)
# ==========================================

def load_mdpfuzz_txt(path):
    """加载 MDPFuzz/Random 的分号分隔 TXT/CSV"""
    try:
        df = pd.read_csv(path, delimiter=';', on_bad_lines='skip', skipinitialspace=True)
        # 处理 Oracle
        if 'Oracle' in df.columns:
            if df['Oracle'].dtype == 'object':
                df['Oracle'] = df['Oracle'].map({'True': True, 'False': False, 'None': None})
            df['is_crash'] = (df['Oracle'] == True)
        elif 'success' in df.columns:
            df['is_crash'] = (df['success'] == False)
        
        # 统一列名
        data = pd.DataFrame({
            'time': pd.to_numeric(df['RunTime'], errors='coerce'),
            'input': df['Input'],
            'is_crash': df['is_crash'],
            'generation': pd.to_numeric(df['Generation'], errors='coerce')
        })
        return data
    except Exception as e:
        print(f"[Loader Error] MDPFuzz/Random ({path}): {e}")
        return None

def load_cure_pkl(path):
    """加载 CureFuzz 的 pickle"""
    try:
        with open(path, 'rb') as f:
            log_data = pickle.load(f)
        
        records = []
        for entry in log_data:
            state = entry.get('mutate_state')
            if state is None: continue
            inp_bytes = state.tobytes() if hasattr(state, 'tobytes') else str(state)
            
            records.append({
                'time': entry.get('elapsed_time'),
                'input': inp_bytes,
                'is_crash': entry.get('did_crash', False),
                'generation': entry.get('parent_depth', 0) + 1
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[Loader Error] CureFuzz ({path}): {e}")
        return None

def load_seq_pkl(path):
    """加载 SeqFuzz 的 pickle"""
    try:
        with open(path, 'rb') as f:
            log_data = pickle.load(f)
            
        records = []
        for entry in log_data:
            state = entry.get('state')
            if state is None: continue
            inp_bytes = state.tobytes() if hasattr(state, 'tobytes') else str(state)
            
            records.append({
                'time': entry.get('timestamp'), 
                'input': inp_bytes,
                'is_crash': entry.get('crashed', False),
                'generation': entry.get('generation', 0)
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[Loader Error] SeqFuzz ({path}): {e}")
        return None

def load_gmodel_pkl(path):
    """加载 G-Model 的 pickle"""
    try:
        with open(path, 'rb') as f:
            log_data = pickle.load(f)
            
        records = []
        for entry in log_data:
            inp = entry.get('input')
            if inp is None: continue
            inp_bytes = tuple(inp) 
            
            records.append({
                'time': entry.get('time'),
                'input': inp_bytes,
                'is_crash': entry.get('is_crash', False),
                'generation': np.nan 
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[Loader Error] G-Model ({path}): {e}")
        return None

def load_qdfuzz_csv(path):
    """加载 QDFuzz 的 CSV"""
    try:
        df = pd.read_csv(path)
        data = pd.DataFrame({
            'time': df['elapsed_time'],
            'input': df['input'],
            'is_crash': df['is_faulty'],
            'generation': df['mutation_count'] if 'mutation_count' in df.columns else np.nan
        })
        return data
    except Exception as e:
        print(f"[Loader Error] QDFuzz ({path}): {e}")
        return None

# 映射类型到加载函数
LOADERS = {
    "mdpfuzz_txt": load_mdpfuzz_txt,
    "cure_pkl": load_cure_pkl,
    "seq_pkl": load_seq_pkl,
    "gmodel_pkl": load_gmodel_pkl,
    "qdfuzz_csv": load_qdfuzz_csv
}

# ==========================================
# 4. 指标计算与主逻辑
# ==========================================

def process_data(label, config):
    path = config["path"]
    loader_type = config["type"]
    
    if not os.path.exists(path):
        print(f"  [跳过] 文件不存在: {path} ({label})")
        return None, None

    # 1. 加载数据
    loader = LOADERS.get(loader_type)
    df = loader(path)
    
    if df is None or df.empty:
        print(f"  [跳过] 数据为空或加载失败: {label}")
        return None, None

    # 2. 时间归一化
    if 'time' not in df.columns or df['time'].isnull().all():
        print(f"  [警告] {label} 缺少有效的时间列。")
        return None, None
        
    start_time = df['time'].min()
    df['norm_time'] = df['time'] - start_time
    
    # 3. 截取前 N 小时
    limit_sec = MAX_HOURS * 3600
    df_period = df[df['norm_time'] <= limit_sec].copy()
    
    # 4. 提取 Unique Crashes
    crashes = df_period[df_period['is_crash'] == True].copy()
    
    if crashes.empty:
        return 0, np.nan # 没崩溃

    # 按时间排序并去重
    crashes = crashes.sort_values('norm_time')
    unique_crashes = crashes.drop_duplicates(subset=['input'], keep='first')
    n_crashes = len(unique_crashes)
    
    # --- 指标 1: Cost (Min / Crash) ---
    if n_crashes > 0:
        time_eff = (MAX_HOURS * 60) / n_crashes
    else:
        time_eff = 0
        
    # --- 指标 2: Avg Generation ---
    avg_gen = np.nan
    if 'generation' in unique_crashes.columns:
        # Random 和 G-Model 通常没有代数，或为 NaN
        gen_vals = unique_crashes['generation'].dropna()
        if not gen_vals.empty and label != "Random": 
            avg_gen = gen_vals.mean()
            
    print(f"  -> {label}: {n_crashes} unique crashes, Cost={time_eff:.1f} min, AvgGen={avg_gen:.1f}")
    return time_eff, avg_gen

def main():
    print(f"--- 开始处理数据 (统计前 {MAX_HOURS} 小时) ---")
    
    metrics_data = {
        "labels": [],
        "time_per_crash": [],
        "gen_avg_depth": [],
        "colors": []
    }
    
    # 遍历配置中的方法 (现在顺序已经是字母序)
    for label, config in FILES_CONFIG.items():
        print(f"正在处理: {label} ...")
        time_eff, avg_gen = process_data(label, config)
        
        # 即使结果是0也加入，以保持图表占位
        if time_eff is not None:
            metrics_data["labels"].append(label)
            metrics_data["time_per_crash"].append(time_eff)
            metrics_data["gen_avg_depth"].append(avg_gen)
            # 使用新的颜色映射
            metrics_data["colors"].append(COLOR_MAP.get(label, "#333333"))

    # ==========================================
    # 5. 绘图 (严格参照 plot-RQ3)
    # ==========================================
    if not metrics_data["labels"]:
        print("没有有效数据用于绘图。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Subplot 1: Time Efficiency (Cost) ---
    ax1 = axes[0]
    bars1 = ax1.bar(metrics_data["labels"], metrics_data["time_per_crash"], 
                    color=metrics_data["colors"], alpha=0.8, edgecolor='black', width=0.6)
    
    ax1.set_title("Time Cost Efficiency", fontweight='bold')
    ax1.set_ylabel("Avg. Minutes per Crash")
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # 数值标签
    for bar in bars1:
        height = bar.get_height()
        label_text = f'{height:.1f}' if height > 0 else 'N/A'
        ax1.text(bar.get_x() + bar.get_width()/2., height, label_text,
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Subplot 2: Average Discovery Generation ---
    ax2 = axes[1]
    
    # 过滤掉 NaN 数据
    valid_indices = [i for i, x in enumerate(metrics_data["gen_avg_depth"]) if not np.isnan(x) and x > 0]
    
    if valid_indices:
        valid_labels = [metrics_data["labels"][i] for i in valid_indices]
        valid_values = [metrics_data["gen_avg_depth"][i] for i in valid_indices]
        valid_colors = [metrics_data["colors"][i] for i in valid_indices]

        bars2 = ax2.bar(valid_labels, valid_values, color=valid_colors, 
                        alpha=0.8, edgecolor='black', width=0.6)
        
        ax2.set_title("Avg. Generation of Discovery", fontweight='bold')
        ax2.set_ylabel("Avg. Generation Index")
        ax2.grid(axis='y', linestyle='--', alpha=0.6)

        # 数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No Generation Data Available", 
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Average Discovery Generation")

    plt.tight_layout()
    output_file = 'RQ3_bw_aligned.png'
    plt.savefig(output_file, dpi=300)
    print(f"\n图表已保存至: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()