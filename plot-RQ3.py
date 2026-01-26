import matplotlib
# [关键修复 1] 强制使用非交互式后端，防止 IDE/Server 环境下绘图冲突导致保存空白
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pickle
import os
import csv
import math

# ================= 配置区域 =================
MAX_H = 12.0  # 统计时限 12小时

# 统一颜色映射
COLOR_MAP = {
    "CureFuzz": "#1f77b4", # Blue
    "G-Model":  "#ff7f0e", # Orange
    "MDPFuzz":  "#2ca02c", # Green
    "QDFuzz":   "#d62728", # Red
    "Random":   "#9467bd", # Purple
    "SeqFuzz":  "#8c564b", # Brown
}

# 绘图顺序
METHODS_ORDER = ["CureFuzz", "G-Model", "MDPFuzz", "QDFuzz", "SeqFuzz"]

# ================= 通用加载工具函数 =================

def safe_load_pickle(filepath):
    if not os.path.exists(filepath):
        # print(f"[Warn] File not found: {filepath}")
        return []
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[Error] loading pickle {filepath}: {e}")
        return []

def safe_load_csv(filepath, delimiter=','):
    if not os.path.exists(filepath):
        # print(f"[Warn] File not found: {filepath}")
        return pd.DataFrame()
    try:
        return pd.read_csv(filepath, delimiter=delimiter, on_bad_lines='skip', skipinitialspace=True)
    except Exception as e:
        print(f"[Error] loading csv {filepath}: {e}")
        return pd.DataFrame()

# ================= Mountain Car (MC) 数据处理 =================
def get_mc_data(base_dir='mc'):
    print(f"--- Loading MC Data from {base_dir} ---")
    data_map = {}
    
    configs = [
        {'label': 'Random',  'file': 'MC_DQN_RT_0_budget730000_logs.txt',      'type': 'mdpfuzz', 'is_random': True},
        {'label': 'MDPFuzz', 'file': 'MC_DQN_NoCov_5_0.01_0.1_0_12h_logs.txt', 'type': 'mdpfuzz', 'is_random': False},
        {'label': 'SeqFuzz', 'file': 'all_run_seeds_0.pkl',                     'type': 'seqfuzz'},
        {'label': 'CureFuzz', 'file': 'selection_log.pkl',                      'type': 'curefuzz'},
        {'label': 'QDFuzz',  'file': 'mc_test_data.csv',                        'type': 'qdfuzz'},
        {'label': 'G-Model', 'file': 'all_test_cases_log.pkl',                  'type': 'gmodel'},
    ]

    for cfg in configs:
        label = cfg['label']
        if label == 'Random': continue
        
        path = os.path.join(base_dir, cfg['file'])
        gens = []
        
        if cfg['type'] == 'curefuzz':
            data = safe_load_pickle(path)
            crashes = [x for x in data if x.get('did_crash', False)]
            crashes.sort(key=lambda x: x.get('elapsed_time', 0))
            seen = set()
            for c in crashes:
                if c.get('elapsed_time', 0) > MAX_H * 3600: continue
                state = c.get('mutate_state')
                if state is None: continue
                key = state.tobytes() if hasattr(state, 'tobytes') else str(state)
                if key not in seen:
                    seen.add(key)
                    gens.append(c.get('parent_depth', 0) + 1)
                    
        elif cfg['type'] == 'gmodel':
            data = safe_load_pickle(path)
            crashes = [x for x in data if x.get('is_crash', False)]
            crashes.sort(key=lambda x: x.get('time', 0))
            seen = set()
            for c in crashes:
                if c.get('time', 0) > MAX_H * 3600: continue
                step = c.get('step', 0)
                if (step % 100) < 50: continue
                inp = c.get('input')
                key = tuple(inp) if isinstance(inp, list) else str(inp)
                if key not in seen:
                    seen.add(key)
                    gens.append(int(step / 100) + 1)
                    
        elif cfg['type'] == 'mdpfuzz':
            try:
                if not os.path.exists(path): continue
                with open(path, 'r') as f:
                    reader = csv.reader(f, delimiter=';')
                    headers = next(reader, None)
                    if headers:
                        headers = [h.strip() for h in headers]
                        rows = list(reader)
                        try:
                            idx_oracle = headers.index('Oracle')
                            idx_gen = headers.index('Generation')
                            idx_rt = headers.index('RunTime')
                            idx_inp = headers.index('Input')
                            
                            rows.sort(key=lambda x: float(x[idx_rt]) if x[idx_rt].strip() != 'None' else 0)
                            
                            start_time = None
                            seen = set()
                            for row in rows:
                                if not row: continue
                                try:
                                    rt = float(row[idx_rt])
                                    if start_time is None: start_time = rt
                                    if (rt - start_time) > MAX_H * 3600: continue
                                    
                                    if row[idx_oracle].strip() == 'True':
                                        gen_val = int(float(row[idx_gen]))
                                        if gen_val == 0: continue 
                                        
                                        inp_str = row[idx_inp].strip()
                                        if inp_str not in seen:
                                            seen.add(inp_str)
                                            gens.append(gen_val)
                                except: continue
                        except ValueError: pass
            except Exception as e:
                print(f"Error parsing MC MDPFuzz: {e}")

        elif cfg['type'] == 'qdfuzz':
            df = safe_load_csv(path)
            if not df.empty:
                seen = set()
                for _, row in df.iterrows():
                    if row.get('is_faulty', False) and row.get('discovery_time', 0) <= MAX_H * 3600:
                        mc_cnt = row.get('mutation_count', 0)
                        if mc_cnt == 0: continue
                        inp = row.get('input')
                        key = str(inp)
                        if key not in seen:
                            seen.add(key)
                            gens.append(int(mc_cnt) + 1)

        elif cfg['type'] == 'seqfuzz':
            data = safe_load_pickle(path)
            crashes = [x for x in data if x.get('crashed', False)]
            seen = set()
            for c in crashes:
                t = c.get('crash_time')
                if t is not None and t > MAX_H * 3600: continue
                gen = c.get('generation', 0)
                if gen == 0: continue
                
                state = c.get('state')
                key = str(state)
                if key not in seen:
                    seen.add(key)
                    gens.append(gen)

        if gens:
            data_map[label] = gens
            
    return data_map

# ================= Bipedal Walker (BW) 数据处理 =================
def get_bw_data(base_dir='bw'):
    print(f"--- Loading BW Data from {base_dir} ---")
    data_map = {}
    
    configs = [
        {'label': 'Random',  'file': 'rt_10_0.01_0.01_0_logs.txt',     'type': 'mdpfuzz', 'is_random': True},
        {'label': 'MDPFuzz', 'file': 'fuzzer_10_0.01_0.01_0_logs.txt', 'type': 'mdpfuzz', 'is_random': False},
        {'label': 'SeqFuzz', 'file': 'all_run_seeds_0.pkl',             'type': 'seqfuzz'},
        {'label': 'CureFuzz', 'file': 'selection_log.pkl',              'type': 'curefuzz'},
        {'label': 'QDFuzz',  'file': '1765639810.5339673_data.csv',     'type': 'qdfuzz'},
        {'label': 'G-Model', 'file': 'all_test_cases_log.pkl',          'type': 'gmodel'},
    ]

    for cfg in configs:
        label = cfg['label']
        if label == 'Random': continue
        
        path = os.path.join(base_dir, cfg['file'])
        gens = []
        
        df = pd.DataFrame()
        if cfg['type'] == 'mdpfuzz':
            df_raw = safe_load_csv(path, delimiter=';')
            if 'RunTime' in df_raw.columns:
                df = pd.DataFrame({
                    'time': pd.to_numeric(df_raw['RunTime'], errors='coerce'),
                    'input': df_raw['Input'],
                    'is_crash': (df_raw['Oracle'].astype(str) == 'True'),
                    'generation': pd.to_numeric(df_raw['Generation'], errors='coerce')
                })
        
        elif cfg['type'] == 'curefuzz':
            raw = safe_load_pickle(path)
            recs = []
            for e in raw:
                recs.append({
                    'time': e.get('elapsed_time'), 
                    'input': str(e.get('mutate_state')),
                    'is_crash': e.get('did_crash', False),
                    'generation': e.get('parent_depth', 0) + 1
                })
            df = pd.DataFrame(recs)
            
        elif cfg['type'] == 'seqfuzz':
            raw = safe_load_pickle(path)
            recs = []
            for e in raw:
                recs.append({
                    'time': e.get('timestamp'),
                    'input': str(e.get('state')),
                    'is_crash': e.get('crashed', False),
                    'generation': e.get('generation', 0)
                })
            df = pd.DataFrame(recs)
            
        elif cfg['type'] == 'gmodel':
            raw = safe_load_pickle(path)
            recs = []
            for e in raw:
                step = e.get('step', 0)
                gen = math.ceil(step / 50.0)
                inp = e.get('input')
                recs.append({
                    'time': e.get('time'),
                    'input': str(inp),
                    'is_crash': e.get('is_crash', False),
                    'generation': gen
                })
            df = pd.DataFrame(recs)
            
        elif cfg['type'] == 'qdfuzz':
            df_raw = safe_load_csv(path)
            if not df_raw.empty:
                df = pd.DataFrame({
                    'time': df_raw['elapsed_time'],
                    'input': df_raw['input'],
                    'is_crash': df_raw['is_faulty'],
                    'generation': df_raw['mutation_count']
                })

        if not df.empty and 'time' in df.columns:
            start_t = df['time'].min()
            df['norm_time'] = df['time'] - start_t
            df = df[df['norm_time'] <= MAX_H * 3600]
            
            crashes = df[df['is_crash'] == True].sort_values('norm_time')
            unique = crashes.drop_duplicates(subset=['input'], keep='first')
            
            valid_gens = unique[unique['generation'] > 0]['generation'].dropna().tolist()
            if valid_gens:
                data_map[label] = valid_gens

    return data_map

# ================= CARLA 数据处理 =================
def get_carla_data(base_dir='carla'):
    print(f"--- Loading CARLA Data from {base_dir} ---")
    data_map = {}
    
    files_config = {
        "curefuzz.csv": {"label": "CureFuzz", "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post", "gen_col": "mutation_generation"},
        "g-model.csv":  {"label": "G-Model", "time_col": "elapsed_time", "special": "g-model", "input_col": "input_post"},
        "mdpfuzz.csv":  {"label": "MDPFuzz", "time_col": "global_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "current_input", "gen_col": "generation"},
        "qdfuzz.csv":   {"label": "QDFuzz", "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post", "gen_col": "mutation_generation"},
        "seqfuzz.csv":  {"label": "SeqFuzz", "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post", "gen_col": "mutation_generation"}
    }

    for fname, cfg in files_config.items():
        label = cfg['label']
        path = os.path.join(base_dir, fname)
        df = safe_load_csv(path)
        if df.empty: continue
        
        if cfg.get('special') == 'g-model':
            df = df.reset_index(drop=True)
            df['generation'] = (df.index // 20) + 1
            gen_col = 'generation'
            start_time = df[cfg['time_col']].min()
            df_filtered = df.copy()
        else:
            gen_col = cfg.get('gen_col')
            target_phase = cfg.get('target_phase')
            if target_phase in df[cfg['phase_col']].values:
                phase_data = df[df[cfg['phase_col']] == target_phase]
                start_time = phase_data[cfg['time_col']].min()
                df_filtered = phase_data.copy()
            else:
                continue

        df_filtered['norm_time'] = df_filtered[cfg['time_col']] - start_time
        df_12h = df_filtered[df_filtered['norm_time'] <= MAX_H * 3600]
        
        is_crash = df_12h['success'].astype(str) == 'False'
        crashes = df_12h[is_crash].copy()
        
        if not crashes.empty:
            crashes = crashes.sort_values('norm_time')
            unique_crashes = crashes.drop_duplicates(subset=[cfg['input_col']], keep='first')
            
            if gen_col in unique_crashes.columns:
                g_list = unique_crashes[gen_col].dropna()
                g_list = g_list[g_list > 0].tolist()
                if g_list:
                    data_map[label] = g_list

    return data_map

# ================= 核心绘图函数 (包含修复) =================

def plot_single_ax(ax, data_map, title):
    """在单个 Axis 上绘制箱线图 + 散点"""
    valid_keys = [m for m in METHODS_ORDER if m in data_map and len(data_map[m]) > 0]
    if not valid_keys:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        ax.set_title(title)
        return

    plot_data = [data_map[k] for k in valid_keys]
    colors = [COLOR_MAP.get(k, '#333') for k in valid_keys]
    
    # 1. 绘制散点 (底层) - [关键修复 2] 添加 rasterized=True 防止 PDF 空白或过大
    for i, (k, vals) in enumerate(zip(valid_keys, plot_data)):
        y_pos = i + 1
        y_jitter = np.random.normal(y_pos, 0.08, size=len(vals))
        ax.scatter(vals, y_jitter, alpha=0.3, # 降低透明度
                   color=COLOR_MAP.get(k), 
                   s=10, edgecolor='none', 
                   zorder=1, # 确保在底层
                   rasterized=True) # 强制栅格化散点图层

    # 2. 绘制箱线图 (顶层)
    box = ax.boxplot(plot_data, vert=False, patch_artist=True,
                     labels=valid_keys, showmeans=True,
                     widths=0.5, showfliers=False, 
                     zorder=10, # 确保在顶层
                     meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6, "zorder":11},
                     medianprops={"color": "black", "linewidth": 1.5, "zorder":11},
                     boxprops={"linewidth": 1, "zorder":10},
                     whiskerprops={"zorder":10},
                     capprops={"zorder":10})
    
    # 上色
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8) # 较高不透明度遮盖散点
        # [关键修复 3] 强制箱体栅格化，防止透明度渲染 Bug
        patch.set_rasterized(True) 

    # 3. 装饰
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xscale('symlog', linthresh=1) # 对数坐标
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, which="both", ls="--", alpha=0.3, zorder=0)
    ax.set_xlabel('Generation (Log Scale)')

def main():
    # 0. 准备工作
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif', 
        'font.size': 12,
        'pdf.fonttype': 42, # 保证字体嵌入
        'ps.fonttype': 42
    })
    
    # 1. 获取数据
    mc_data = get_mc_data(base_dir='mc')
    bw_data = get_bw_data(base_dir='bw')
    carla_data = get_carla_data(base_dir='carla')
    
    # 2. 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    
    # 3. 分别绘图
    plot_single_ax(axes[0], mc_data, "Mountain Car")
    plot_single_ax(axes[1], bw_data, "Bipedal Walker")
    plot_single_ax(axes[2], carla_data, "CARLA")
    
    plt.tight_layout()
    
    # 4. 保存
    save_name = 'RQ3_Combined_Boxplot.pdf'
    # 使用 fig.savefig 确保完整保存
    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Combined plot saved to {save_name} (Rasterized optimization applied).")

if __name__ == "__main__":
    main()