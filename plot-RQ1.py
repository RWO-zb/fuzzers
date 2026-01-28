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

# [修改重点] 重新增大字号，并配合紧凑画布
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 22,             # [回调] 16 -> 22 (对抗 LaTeX 缩放)
    'axes.labelsize': 24,        # [回调] 轴标签更大
    'axes.titlesize': 26,        
    'xtick.labelsize': 22,       
    'ytick.labelsize': 22,       
    'legend.fontsize': 20,       
    'lines.linewidth': 3.0       # 线条加粗以匹配大字体
})

COLORS = {
    'Random':      '#95a5a6',  
    'MDPFuzz':     '#e74c3c',  
    'SeqDivFuzz':  '#2ecc71',  
    'CureFuzz':    '#9b59b6', 
    'QDFuzz':      '#f39c12',  
    'G-Model':     '#3498db',  
}

MAX_H = 12.0
VIEW_LIMIT_H = 13.5 # [修改]稍微加宽 X 轴视野，给右侧大字体标签留空间
MARKERS_X_H = np.arange(2, MAX_H + 0.1, 2)

# ==========================================
# 2. 通用解析工具函数
# ==========================================
def load_pickle(filepath):
    if not os.path.exists(filepath): return None
    try:
        with open(filepath, 'rb') as f: return pickle.load(f)
    except: return None

# ==========================================
# 3. 数据加载函数 (保持逻辑不变)
# ==========================================
def get_mc_data(base_dir='mc'):
    data_map = {}
    configs = [
        {'label': 'Random',      'file': 'MC_DQN_RT_0_budget730000_logs.txt', 'type': 'mdpfuzz'},
        {'label': 'MDPFuzz',     'file': 'MC_DQN_NoCov_5_0.01_0.1_0_12h_logs.txt', 'type': 'mdpfuzz'},
        {'label': 'SeqDivFuzz',  'file': 'all_run_seeds_0.pkl', 'type': 'seqfuzz'},
        {'label': 'CureFuzz',    'file': 'selection_log.pkl', 'type': 'curefuzz'},
        {'label': 'QDFuzz',      'file': 'mc_test_data.csv', 'type': 'qdfuzz'},
        {'label': 'G-Model',     'file': 'all_test_cases_log.pkl', 'type': 'gmodel'},
    ]
    for cfg in configs:
        path = os.path.join(base_dir, cfg['file'])
        label = cfg['label']
        times = np.array([])
        if cfg['type'] == 'mdpfuzz' and os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=';', skipinitialspace=True, on_bad_lines='skip')
                df.columns = [c.strip() for c in df.columns]
                if len(df) > 10000: df = df.iloc[10000:].copy() 
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
        
        elif cfg['type'] == 'qdfuzz' and os.path.exists(path):
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

def get_carla_data(base_dir='carla'):
    data_map = {}
    files_config = {
        "curefuzz.csv": {"label": "CureFuzz",   "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"},
        "g-model.csv":  {"label": "G-Model",    "time_col": "elapsed_time", "special": "g-model", "input_col": "input_post"},
        "mdpfuzz.csv":  {"label": "MDPFuzz",    "time_col": "global_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "current_input"},
        "qdfuzz.csv":   {"label": "QDFuzz",     "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"},
        "random.csv":   {"label": "Random",     "time_col": "global_time", "special": "random", "input_col": "current_input"},
        "seqfuzz.csv":  {"label": "SeqDivFuzz", "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post"}
    }
    for fname, cfg in files_config.items():
        path = os.path.join(base_dir, fname)
        label = cfg['label']
        times = np.array([])
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "special" in cfg and cfg["special"] == "g-model":
                    start_time = df[df['method'] == 'generative+novelty'][cfg['time_col']].min() if 'generative+novelty' in df['method'].values else df[cfg['time_col']].min()
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
                    else: df_filtered = pd.DataFrame()
                if not df_filtered.empty:
                    df_filtered['norm_time'] = df_filtered[cfg['time_col']] - start_time
                    is_crash = (df_filtered['success'] == False) if df_filtered['success'].dtype == 'bool' else (df_filtered['success'].astype(str) == 'False')
                    crashes = df_filtered[is_crash].copy()
                    input_col = cfg.get("input_col")
                    if input_col and input_col in crashes.columns:
                        crashes = crashes.sort_values('norm_time')
                        crashes = crashes.drop_duplicates(subset=[input_col], keep='first')
                        times = np.sort(crashes['norm_time'].values)
            except Exception as e: print(f"Error CARLA {label}: {e}")
        data_map[label] = times
    return data_map

def get_bw_data(base_dir='bw'):
    data_map = {}
    configs = [
        {'label': 'Random',      'file': 'rt_10_0.01_0.01_0_logs.txt', 'type': 'mdpfuzz'},
        {'label': 'MDPFuzz',     'file': 'fuzzer_10_0.01_0.01_0_logs.txt', 'type': 'mdpfuzz'},
        {'label': 'SeqDivFuzz',  'file': 'all_run_seeds_0.pkl', 'type': 'seqfuzz'},
        {'label': 'CureFuzz',    'file': 'selection_log.pkl', 'type': 'curefuzz'},
        {'label': 'QDFuzz',      'file': '1765639810.5339673_data.csv', 'type': 'qdfuzz'},
        {'label': 'G-Model',     'file': 'all_test_cases_log.pkl', 'type': 'gmodel'},
    ]
    for cfg in configs:
        path = os.path.join(base_dir, cfg['file'])
        label = cfg['label']
        times = np.array([])
        if cfg['type'] == 'mdpfuzz' and os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=';', on_bad_lines='skip', skipinitialspace=True)
                if df.shape[1] < 2: df = pd.read_csv(path, sep=',', on_bad_lines='skip', skipinitialspace=True)
                df.columns = [c.strip() for c in df.columns]
                if 'Oracle' in df.columns:
                    df['is_crash'] = df['Oracle'].astype(str).str.strip() == 'True'
                    if 'RunTime' in df.columns:
                        df['RunTime'] = pd.to_numeric(df['RunTime'], errors='coerce')
                        crashes = df[df['is_crash'] == True].copy()
                        if 'Input' in crashes.columns: crashes = crashes.drop_duplicates(subset=['Input'], keep='first')
                        if not crashes.empty:
                            start_time = df['RunTime'].min()
                            times = np.sort(crashes['RunTime'] - start_time)
            except: pass
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
        elif cfg['type'] == 'qdfuzz' and os.path.exists(path):
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
                        if ti not in seen: seen[ti] = e.get('time', 0)
                times = np.sort(list(seen.values()))
        data_map[label] = times
    return data_map

# ==========================================
# 6. 绘图核心函数
# ==========================================
def plot_subplot(ax, data_map):
    final_labels = []
    max_data_y = 0
    for label, times in data_map.items():
        color = COLORS.get(label, '#333333')
        if len(times) > 0:
            limit_sec = MAX_H * 3600
            times = times[times <= limit_sec]
            times_h = times / 3600.0
        else: times_h = np.array([])
        
        if len(times_h) > 0:
            x_plot = np.concatenate(([0], times_h))
            y_plot = np.concatenate(([0], np.arange(1, len(times_h) + 1)))
            ax.step(x_plot, y_plot, where='post', label=label, color=color, alpha=0.9)
            last_y_val = y_plot[-1]
            max_data_y = max(max_data_y, last_y_val)
            final_labels.append({'label': label, 'x': x_plot[-1], 'y': last_y_val, 'color': color})
            last_crash_time = times_h[-1]
        else: last_crash_time = 0
        
        valid_markers_x, valid_markers_y = [], []
        if len(times_h) > 0:
            for mx in MARKERS_X_H:
                if mx <= last_crash_time:
                    count = np.searchsorted(times_h, mx, side='right')
                    valid_markers_x.append(mx)
                    valid_markers_y.append(count)
        if valid_markers_x:
            ax.plot(valid_markers_x, valid_markers_y, linestyle='none', marker='^', 
                     color=color, markersize=10, markeredgecolor='white', markeredgewidth=1.5)

    # --- 强化的防重叠逻辑 ---
    max_label_y = max_data_y
    if final_labels:
        final_labels.sort(key=lambda k: k['y'])
        
        # [修改] 显著增加最小间距 (12% of range)，确保大字体不重叠
        range_y = max_data_y if max_data_y > 0 else 1.0
        min_dist = max(range_y * 0.12, 1.5) 
        
        for i in range(1, len(final_labels)):
            prev, curr = final_labels[i-1], final_labels[i]
            if curr['y'] - prev['y'] < min_dist: 
                curr['y'] = prev['y'] + min_dist
        
        max_label_y = final_labels[-1]['y']
        
        for item in final_labels:
            ax.text(item['x'] + 0.2, item['y'], item['label'], 
                    color=item['color'], fontsize=22, # [回调] 大字体
                    verticalalignment='center', fontweight='bold')

    # [修改] 顶部预留更多空间 (1.25倍)，防止最大号字体的标签被切
    top_limit = max(max_data_y, max_label_y) * 1.25
    ax.set_ylim(0, top_limit)
    
    # [修改] 稍微放宽 X 轴右侧视野，给最右侧的 "SeqDivFuzz" 等长标签留出位置
    ax.set_xlim(0, VIEW_LIMIT_H)
    ax.set_xticks(np.arange(0, 13, 2))
    ax.set_xlabel("Time (h)", fontsize=24)
    ax.set_ylabel("Unique Crashes", fontsize=24)
    ax.grid(True, linestyle='--', alpha=0.5, color='#95a5a6')

def main():
    print("Initializing plots...")
    
    # [修改] 使用紧凑但不过小的画布 (7, 5.5)
    # 结合 fontsize=22，这会让图片在 Overleaf 缩小后依然显得字很大
    single_figsize = (7, 5.5) 

    print("--- Processing Mountain Car ---")
    mc_data = get_mc_data('mc')
    fig1, ax1 = plt.subplots(figsize=single_figsize)
    plot_subplot(ax1, mc_data) 
    plt.tight_layout()
    fig1.savefig('RQ1_MountainCar.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("--- Processing BipedalWalker ---")
    bw_data = get_bw_data('bw')
    fig2, ax2 = plt.subplots(figsize=single_figsize)
    plot_subplot(ax2, bw_data)
    plt.tight_layout()
    fig2.savefig('RQ1_BipedalWalker.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("--- Processing CARLA ---")
    carla_data = get_carla_data('carla')
    fig3, ax3 = plt.subplots(figsize=single_figsize)
    plot_subplot(ax3, carla_data)
    plt.tight_layout()
    fig3.savefig('RQ1_CARLA.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("Done.")

if __name__ == "__main__":
    main()