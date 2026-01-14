import os
import pickle
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 全局样式与颜色配置
# ==========================================
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2
})

COLORS = {
    'Random':   '#7f8c8d',  # 灰色
    'MDPFuzz':  '#e74c3c',  # 红色
    'SeqFuzz':  '#2ecc71',  # 绿色
    'CureFuzz': '#9b59b6',  # 紫色
    'QDFuzz':   '#f39c12',  # 橙色
    'G-Model':  '#3498db',  # 蓝色
}

MAX_H = 12.0
VIEW_LIMIT_H = 12.5
MARKERS_X_H = np.arange(2, MAX_H + 0.1, 2)

# ==========================================
# 2. 通用解析工具函数
# ==========================================
def load_pickle(filepath):
    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[Error] Load pickle {filepath}: {e}")
        return None

# ==========================================
# 3. Mountain Car (MC) 数据加载逻辑
# ==========================================
def get_mc_data(base_dir='mc'):
    data_map = {}
    
    configs = [
        {'label': 'Random', 'file': 'MC_DQN_RT_0_budget730000_logs.txt', 'type': 'mdpfuzz'},
        {'label': 'MDPFuzz', 'file': 'MC_DQN_NoCov_5_0.01_0.1_0_12h_logs.txt', 'type': 'mdpfuzz'},
        {'label': 'SeqFuzz', 'file': 'all_run_seeds_0.pkl', 'type': 'seqfuzz'},
        {'label': 'CureFuzz', 'file': 'selection_log.pkl', 'type': 'curefuzz'},
        {'label': 'QDFuzz', 'file': 'mc_test_data.csv', 'type': 'qdfuzz'},
        {'label': 'G-Model', 'file': 'all_test_cases_log.pkl', 'type': 'gmodel'},
    ]

    for cfg in configs:
        path = os.path.join(base_dir, cfg['file'])
        label = cfg['label']
        times = np.array([])

        if cfg['type'] == 'mdpfuzz':
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, sep=';', skipinitialspace=True, on_bad_lines='skip')
                    df.columns = [c.strip() for c in df.columns]
                    if len(df) > 10000: df = df.iloc[10000:].copy() # MC specific skip
                    if 'Oracle' in df.columns:
                        df['Oracle'] = df['Oracle'].astype(str).str.strip() == 'True'
                        crashes = df[df['Oracle'] == True].copy()
                        if not crashes.empty and 'RunTime' in crashes.columns:
                            crashes['RunTime'] = pd.to_numeric(crashes['RunTime'], errors='coerce')
                            start_time = df['RunTime'].min()
                            crashes['CrashTime'] = crashes['RunTime'] - start_time
                            if 'Input' in crashes.columns: crashes = crashes.drop_duplicates(subset=['Input'])
                            times = np.sort(crashes['CrashTime'].dropna().values)
                except: pass

        elif cfg['type'] == 'seqfuzz':
            data = load_pickle(path)
            if data:
                seen, temp_t = set(), []
                for e in data:
                    s = e.get('state')
                    if s is None: s = e.get('mutate_state')
                    if s is None: continue
                    try: sb = s.tobytes() if hasattr(s,'tobytes') else bytes(s)
                    except: continue
                    if sb not in seen:
                        seen.add(sb)
                        if e.get('crashed') or e.get('did_crash'):
                            t = e.get('crash_time')
                            if t: temp_t.append(t)
                times = np.sort(temp_t)

        elif cfg['type'] == 'curefuzz':
            data = load_pickle(path)
            if data:
                seen, temp_t = set(), []
                for e in data:
                    s = e.get('mutate_state')
                    if s is None: continue
                    try: sb = s.tobytes()
                    except: continue
                    if sb not in seen:
                        seen.add(sb)
                        if e.get('did_crash'):
                            t = e.get('crash_time') or e.get('elapsed_time')
                            if t: temp_t.append(t)
                times = np.sort(temp_t)
        
        elif cfg['type'] == 'qdfuzz':
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    temp_t, seen = [], set()
                    for _, row in df.iterrows():
                        if not row.get('is_faulty'): continue
                        inp = row.get('input')
                        try: 
                            if isinstance(inp, str): inp = ast.literal_eval(inp)
                            sb = np.array(inp, dtype=np.float32).tobytes()
                            if sb not in seen:
                                seen.add(sb)
                                if row.get('discovery_time'): temp_t.append(row['discovery_time'])
                        except: pass
                    times = np.sort(temp_t)
                except: pass

        elif cfg['type'] == 'gmodel':
            data = load_pickle(path)
            if data:
                seen, temp_t = set(), []
                for e in data:
                    if e.get('is_crash'):
                        try:
                            ti = tuple(e['input'])
                            if ti not in seen:
                                seen.add(ti)
                                temp_t.append(e.get('timestamp', 0))
                        except: pass
                times = np.sort(temp_t)

        data_map[label] = times
        
    return data_map

# ==========================================
# 4. CARLA 数据加载逻辑
# ==========================================
def get_carla_data(base_dir='carla'):
    data_map = {}
    
    files_config = {
        "curefuzz.csv": {"label": "CureFuzz", "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"},
        "g-model.csv":  {"label": "G-Model", "time_col": "elapsed_time", "special": "g-model", "input_col": "input_post"},
        "mdpfuzz.csv":  {"label": "MDPFuzz", "time_col": "global_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "current_input"},
        "qdfuzz.csv":   {"label": "QDFuzz", "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"},
        "random.csv":   {"label": "Random", "time_col": "global_time", "special": "random", "input_col": "current_input"},
        "seqfuzz.csv":  {"label": "SeqFuzz", "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"}
    }

    for fname, cfg in files_config.items():
        path = os.path.join(base_dir, fname)
        label = cfg['label']
        times = np.array([])
        
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "special" in cfg and cfg["special"] == "g-model":
                    if 'generative+novelty' in df['method'].values:
                        start_time = df[df['method'] == 'generative+novelty'][cfg['time_col']].min()
                    else:
                        start_time = df[cfg['time_col']].min()
                    df_filtered = df[df[cfg['time_col']] >= start_time].copy()
                elif "special" in cfg and cfg["special"] == "random":
                    start_time = df[cfg['time_col']].min()
                    df_filtered = df.copy()
                else:
                    target_phase = cfg["target_phase"]
                    if target_phase in df[cfg['phase_col']].values:
                        phase_data = df[df[cfg['phase_col']] == target_phase]
                        start_time = phase_data[cfg['time_col']].min()
                        df_filtered = phase_data.copy()
                    else:
                        df_filtered = pd.DataFrame()
                
                if not df_filtered.empty:
                    df_filtered['norm_time'] = df_filtered[cfg['time_col']] - start_time
                    if df_filtered['success'].dtype == 'bool': is_crash = df_filtered['success'] == False
                    else: is_crash = df_filtered['success'].astype(str) == 'False'
                    crashes = df_filtered[is_crash].copy()
                    
                    input_col = cfg.get("input_col")
                    if input_col and input_col in crashes.columns:
                        crashes = crashes.sort_values('norm_time')
                        crashes = crashes.drop_duplicates(subset=[input_col], keep='first')
                        times = np.sort(crashes['norm_time'].values)
            except Exception as e:
                print(f"Error CARLA {label}: {e}")
        
        data_map[label] = times
    return data_map

# ==========================================
# 5. BipedalWalker (BW) 数据加载逻辑
# ==========================================
def get_bw_data(base_dir='bw'):
    data_map = {}
    
    configs = [
        {'label': 'Random', 'file': 'rt_10_0.01_0.01_0_logs.txt', 'type': 'mdpfuzz'},
        {'label': 'MDPFuzz', 'file': 'fuzzer_10_0.01_0.01_0_logs.txt', 'type': 'mdpfuzz'},
        {'label': 'SeqFuzz', 'file': 'all_run_seeds_0.pkl', 'type': 'seqfuzz'},
        {'label': 'CureFuzz', 'file': 'selection_log.pkl', 'type': 'curefuzz'},
        {'label': 'QDFuzz', 'file': '1765639810.5339673_data.csv', 'type': 'qdfuzz'},
        {'label': 'G-Model', 'file': 'all_test_cases_log.pkl', 'type': 'gmodel'},
    ]
    
    for cfg in configs:
        path = os.path.join(base_dir, cfg['file'])
        label = cfg['label']
        times = np.array([])
        
        # --- BW Specific Parsers ---
        if cfg['type'] == 'mdpfuzz':
            if os.path.exists(path):
                try:
                    # 1. 尝试读取 (更鲁棒的读取方式)
                    df = pd.read_csv(path, sep=';', on_bad_lines='skip', skipinitialspace=True)
                    # 如果读取结果只有1列，可能分隔符是逗号
                    if df.shape[1] < 2:
                        df = pd.read_csv(path, sep=',', on_bad_lines='skip', skipinitialspace=True)
                    
                    # 2. 清理列名 (去除前后空格)
                    df.columns = [c.strip() for c in df.columns]

                    if 'Oracle' in df.columns:
                        # 3. 更鲁棒的布尔转换：转字符串 -> 去空格 -> 比较
                        df['is_crash'] = df['Oracle'].astype(str).str.strip() == 'True'
                        
                        if 'RunTime' in df.columns:
                            df['RunTime'] = pd.to_numeric(df['RunTime'], errors='coerce')
                            
                            # 4. 筛选 Crash
                            crashes = df[df['is_crash'] == True].copy()
                            
                            # 5. 去重 (Input列)
                            if 'Input' in crashes.columns:
                                crashes = crashes.drop_duplicates(subset=['Input'], keep='first')
                            
                            # 6. 计算时间
                            if not crashes.empty:
                                start_time = df['RunTime'].min()
                                times = np.sort(crashes['RunTime'] - start_time)
                except Exception as e:
                    print(f"Error parsing BW MDPFuzz {label}: {e}")

        elif cfg['type'] == 'seqfuzz':
            data = load_pickle(path)
            if data:
                seen, temp_t = set(), []
                for e in data:
                    s = e.get('state')
                    if s is None: continue
                    try: sb = s.tobytes()
                    except: continue
                    if len(sb) not in [60, 120]: continue
                    if sb not in seen:
                        seen.add(sb)
                        if e.get('crashed'):
                            t = e.get('timestamp')
                            if t: temp_t.append(t)
                if temp_t:
                    temp_t.sort()
                    start = min(temp_t) if temp_t[0] > 1e9 else 0
                    times = np.array([t - start for t in temp_t]) if start > 0 else np.array(temp_t)

        elif cfg['type'] == 'curefuzz':
            data = load_pickle(path)
            if data:
                seen, temp_t = set(), []
                for e in data:
                    s = e.get('mutate_state')
                    if s is None: continue
                    try: sb = s.tobytes()
                    except: continue
                    if len(sb) not in [60, 120]: continue
                    if sb not in seen:
                        seen.add(sb)
                        if e.get('did_crash'):
                            t = e.get('elapsed_time')
                            if t: temp_t.append(t)
                times = np.sort(temp_t)

        elif cfg['type'] == 'qdfuzz':
             if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if 'is_faulty' in df.columns:
                        if df['is_faulty'].dtype == 'object': df['is_faulty'] = df['is_faulty'].astype(str).str.lower() == 'true'
                        else: df['is_faulty'] = df['is_faulty'].astype(bool)
                        df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce').fillna(0)
                        unique = df.dropna(subset=['input']).drop_duplicates(subset=['input'], keep='first')
                        crashes = unique[unique['is_faulty'] == True]
                        times = np.sort(crashes['elapsed_time'].tolist())
                except: pass

        elif cfg['type'] == 'gmodel':
            data = load_pickle(path)
            if data:
                seen, temp_t = {}, []
                for e in data:
                    if e.get('is_crash'):
                        ti = tuple(e['input'])
                        if ti not in seen:
                            seen[ti] = e.get('time', 0)
                times = np.sort(list(seen.values()))

        data_map[label] = times
    return data_map

# ==========================================
# 6. 绘图核心函数 (统一逻辑)
# ==========================================
def plot_subplot(ax, data_map, title):
    for label, times in data_map.items():
        color = COLORS.get(label, '#333333')
        
        # 截断数据 (<= 12h)
        if len(times) > 0:
            limit_sec = MAX_H * 3600
            times = times[times <= limit_sec]
            times_h = times / 3600.0
        else:
            times_h = np.array([])
        
        # Step Plot
        if len(times_h) > 0:
            x_plot = np.concatenate(([0], times_h))
            y_plot = np.concatenate(([0], np.arange(1, len(times_h) + 1)))
            ax.step(x_plot, y_plot, where='post', label=label, color=color)
            last_crash_time = times_h[-1]
        else:
            last_crash_time = 0
            # 不画线，但保留循环以处理 potential
        
        # Markers
        valid_markers_x = []
        valid_markers_y = []
        
        if len(times_h) > 0:
            for mx in MARKERS_X_H:
                if mx <= last_crash_time:
                    count = np.searchsorted(times_h, mx, side='right')
                    valid_markers_x.append(mx)
                    valid_markers_y.append(count)
        
        if valid_markers_x:
            ax.plot(valid_markers_x, valid_markers_y, linestyle='none', marker='^', 
                     color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)

    ax.set_xlim(0, VIEW_LIMIT_H)
    ax.set_xticks(np.arange(0, 13, 2))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Unique Crashes")
    ax.set_title(title)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

# ==========================================
# 7. 主程序
# ==========================================
def main():
    print("Initializing plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    print("--- Loading Mountain Car Data ---")
    mc_data = get_mc_data('mc')
    plot_subplot(ax1, mc_data, "MountainCar")
    
    print("--- Loading CARLA Data ---")
    carla_data = get_carla_data('carla')
    plot_subplot(ax2, carla_data, "CARLA")
    
    print("--- Loading BipedalWalker Data ---")
    bw_data = get_bw_data('bw')
    plot_subplot(ax3, bw_data, "BipedalWalker")
    
    plt.tight_layout()
    output_file = 'RQ1.png'
    try:
        plt.savefig(output_file, dpi=300)
        print(f"\n[Success] Combined chart saved to: {output_file}")
    except Exception as e:
        print(f"\n[Error] Failed to save chart: {e}")

if __name__ == "__main__":
    main()