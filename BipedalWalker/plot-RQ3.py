import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json

# ==========================================
# 1. 全局配置与风格设置 (参照 plot-RQ3)
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

# 颜色映射 (确保每个方法有固定的颜色)
COLOR_MAP = {
    "MDPFuzz":  "#d62728", # 红色
    "Random":   "#7f7f7f", # 灰色
    "CureFuzz": "#9467bd", # 紫色
    "SeqFuzz":  "#2ca02c", # 绿色
    "QDFuzz":   "#1f77b4", # 蓝色
    "G-Model":  "#ff7f0e", # 橙色
}

# 统计截断时间 (小时)
MAX_HOURS = 12.0

# ==========================================
# 2. 文件路径配置 (请在此处修改你的文件名)
# ==========================================
# 注意：type 字段决定了使用哪种加载逻辑
FILES_CONFIG = {
    "MDPFuzz": {
        "path": "fuzzer_10_0.01_0.01_0_logs.txt", 
        "type": "mdpfuzz_txt" 
    },
    "Random": {
        "path": "rt_10_0.01_0.01_0_logs.txt",          
        "type": "mdpfuzz_txt"  # Random 通常格式同 MDPFuzz
    },
    "CureFuzz": {
        "path": "selection_log.pkl",               
        "type": "cure_pkl"
    },
    "SeqFuzz": {
        "path": "all_run_seeds_0.pkl",             
        "type": "seq_pkl"
    },
    "G-Model": {
        "path": "all_test_cases_log.pkl",          
        "type": "gmodel_pkl"
    },
    "QDFuzz": {
        "path": "1765639810.5339673_data.csv",                 
        "type": "qdfuzz_csv"
    }
}

# ==========================================
# 3. 数据加载器 (集成各脚本逻辑)
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
            # CureFuzz 逻辑: input -> mutate_state
            state = entry.get('mutate_state')
            if state is None: continue
            # 将 numpy array 转为 bytes 以便去重
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
            # SeqFuzz 逻辑: input -> state (需要转bytes)
            state = entry.get('state')
            if state is None: continue
            inp_bytes = state.tobytes() if hasattr(state, 'tobytes') else str(state)
            
            records.append({
                'time': entry.get('timestamp'), # 假设 timestamp 是累计时间
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
            # G-Model 逻辑: input -> input (通常是list)
            inp = entry.get('input')
            if inp is None: continue
            inp_bytes = tuple(inp) # list 转 tuple 以便去重
            
            records.append({
                'time': entry.get('time'),
                'input': inp_bytes,
                'is_crash': entry.get('is_crash', False),
                'generation': np.nan # G-Model 通常没有代数概念
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[Loader Error] G-Model ({path}): {e}")
        return None

def load_qdfuzz_csv(path):
    """加载 QDFuzz 的 CSV"""
    try:
        df = pd.read_csv(path)
        # QDFuzz 逻辑: elapsed_time, is_faulty, input, mutation_count
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
        if not gen_vals.empty and label != "Random": # Random 代数无意义，强制 NaN
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
    
    # 遍历配置中的方法
    for label, config in FILES_CONFIG.items():
        print(f"正在处理: {label} ...")
        time_eff, avg_gen = process_data(label, config)
        
        # 只要文件处理没报错（即使结果是0），都加入图表以便对比
        if time_eff is not None:
            metrics_data["labels"].append(label)
            metrics_data["time_per_crash"].append(time_eff)
            metrics_data["gen_avg_depth"].append(avg_gen)
            # 使用预定义的颜色，如果未定义则默认为黑色
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
    
    # 过滤掉 NaN 数据 (Random/G-Model 可能没有代数)
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
    output_file = 'RQ3_bw.png'
    plt.savefig(output_file, dpi=300)
    print(f"\n图表已保存至: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()