import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pickle
import os
import csv
import math


MAX_H = 12.0  
COLOR_MAP = {
    "CureFuzz":   "#1f77b4", "G-Model":    "#ff7f0e", "MDPFuzz":    "#2ca02c",
    "QDFuzz":     "#d62728", "Random":     "#9467bd", "SeqDivFuzz": "#8c564b",
}
METHODS_ORDER = ["MDPFuzz", "CureFuzz", "G-Model", "SeqDivFuzz", "QDFuzz"]

def safe_load_pickle(filepath):
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'rb') as f: return pickle.load(f)
    except: return []

def safe_load_csv(filepath, delimiter=','):
    if not os.path.exists(filepath): return pd.DataFrame()
    try: return pd.read_csv(filepath, delimiter=delimiter, on_bad_lines='skip', skipinitialspace=True)
    except: return pd.DataFrame()

def get_mc_data(base_dir='mc'):
    print(f"--- Loading MC Data from {base_dir} ---")
    data_map = {}
    configs = [
        {'label': 'Random',     'file': 'MC_DQN_RT_0_budget730000_logs.txt',      'type': 'mdpfuzz', 'is_random': True},
        {'label': 'MDPFuzz',    'file': 'MC_DQN_NoCov_5_0.01_0.1_0_12h_logs.txt', 'type': 'mdpfuzz', 'is_random': False},
        {'label': 'SeqDivFuzz', 'file': 'all_run_seeds_0.pkl',                    'type': 'seqfuzz'},
        {'label': 'CureFuzz',   'file': 'selection_log.pkl',                      'type': 'curefuzz'},
        {'label': 'QDFuzz',     'file': 'mc_test_data.csv',                       'type': 'qdfuzz'},
        {'label': 'G-Model',    'file': 'all_test_cases_log.pkl',                 'type': 'gmodel'},
    ]
    for cfg in configs:
        label = cfg['label']
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
                            idx_oracle, idx_gen, idx_rt, idx_inp = headers.index('Oracle'), headers.index('Generation'), headers.index('RunTime'), headers.index('Input')
                            rows.sort(key=lambda x: float(x[idx_rt]) if x[idx_rt].strip() != 'None' else 0)
                            start_time, seen = None, set()
                            for row in rows:
                                if not row: continue
                                try:
                                    rt = float(row[idx_rt])
                                    if start_time is None: start_time = rt
                                    if (rt - start_time) > MAX_H * 3600: continue
                                    if row[idx_oracle].strip() == 'True':
                                        gen_val = int(float(row[idx_gen]))
                                        if gen_val == 0 and not cfg.get('is_random', False): continue 
                                        inp_str = row[idx_inp].strip()
                                        if inp_str not in seen:
                                            seen.add(inp_str)
                                            gens.append(max(1, gen_val))
                                except: continue
                        except ValueError: pass
            except: pass
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
        if gens: data_map[label] = gens
    return data_map

def get_bw_data(base_dir='bw'):
    print(f"--- Loading BW Data from {base_dir} ---")
    data_map = {}
    configs = [
        {'label': 'Random',     'file': 'rt_10_0.01_0.01_0_logs.txt',     'type': 'mdpfuzz', 'is_random': True},
        {'label': 'MDPFuzz',    'file': 'fuzzer_10_0.01_0.01_0_logs.txt', 'type': 'mdpfuzz', 'is_random': False},
        {'label': 'SeqDivFuzz', 'file': 'all_run_seeds_0.pkl',            'type': 'seqfuzz'},
        {'label': 'CureFuzz',   'file': 'selection_log.pkl',              'type': 'curefuzz'},
        {'label': 'QDFuzz',     'file': '1765639810.5339673_data.csv',    'type': 'qdfuzz'},
        {'label': 'G-Model',    'file': 'all_test_cases_log.pkl',         'type': 'gmodel'},
    ]
    for cfg in configs:
        label = cfg['label']
        path = os.path.join(base_dir, cfg['file'])
        gens, df = [], pd.DataFrame()
        if cfg['type'] == 'mdpfuzz':
            df_raw = safe_load_csv(path, delimiter=';')
            if 'RunTime' in df_raw.columns:
                df = pd.DataFrame({'time': pd.to_numeric(df_raw['RunTime'], errors='coerce'), 'input': df_raw['Input'], 'is_crash': (df_raw['Oracle'].astype(str) == 'True'), 'generation': pd.to_numeric(df_raw['Generation'], errors='coerce')})
        elif cfg['type'] == 'curefuzz':
            raw = safe_load_pickle(path)
            recs = [{'time': e.get('elapsed_time'), 'input': str(e.get('mutate_state')), 'is_crash': e.get('did_crash', False), 'generation': e.get('parent_depth', 0) + 1} for e in raw]
            df = pd.DataFrame(recs)
        elif cfg['type'] == 'seqfuzz':
            raw = safe_load_pickle(path)
            recs = [{'time': e.get('timestamp'), 'input': str(e.get('state')), 'is_crash': e.get('crashed', False), 'generation': e.get('generation', 0)} for e in raw]
            df = pd.DataFrame(recs)
        elif cfg['type'] == 'gmodel':
            raw = safe_load_pickle(path)
            recs = [{'time': e.get('time'), 'input': str(e.get('input')), 'is_crash': e.get('is_crash', False), 'generation': math.ceil(e.get('step', 0)/50.0)} for e in raw]
            df = pd.DataFrame(recs)
        elif cfg['type'] == 'qdfuzz':
            df_raw = safe_load_csv(path)
            if not df_raw.empty:
                df = pd.DataFrame({'time': df_raw['elapsed_time'], 'input': df_raw['input'], 'is_crash': df_raw['is_faulty'], 'generation': df_raw['mutation_count']})
        if not df.empty and 'time' in df.columns:
            start_t = df['time'].min()
            df['norm_time'] = df['time'] - start_t
            df = df[df['norm_time'] <= MAX_H * 3600]
            crashes = df[df['is_crash'] == True].sort_values('norm_time')
            unique = crashes.drop_duplicates(subset=['input'], keep='first')
            valid_gens = unique[unique['generation'] >= 0]['generation'].dropna().tolist()
            valid_gens = [max(1, g) for g in valid_gens]
            if valid_gens: data_map[label] = valid_gens
    return data_map

def get_carla_data(base_dir='carla'):
    print(f"--- Loading CARLA Data from {base_dir} ---")
    data_map = {}
    files_config = {
        "random.csv": {"label": "Random", "time_col": "global_time", "phase_col": "phase", "target_phase": "RT", "input_col": "current_input", "gen_col": "generation"},
        "curefuzz.csv": {"label": "CureFuzz", "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post", "gen_col": "mutation_generation"},
        "g-model.csv":  {"label": "G-Model", "time_col": "elapsed_time", "special": "g-model", "input_col": "input_post"},
        "mdpfuzz.csv":  {"label": "MDPFuzz", "time_col": "global_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "current_input", "gen_col": "generation"},
        "qdfuzz.csv":   {"label": "QDFuzz", "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post", "gen_col": "mutation_generation"},
        "seqfuzz.csv":  {"label": "SeqDivFuzz", "time_col": "elapsed_time", "phase_col": "phase", "target_phase": "Phase2", "input_col": "input_post", "gen_col": "mutation_generation"}
    }
    for fname, cfg in files_config.items():
        label = cfg['label']
        path = os.path.join(base_dir, fname)
        df = safe_load_csv(path)
        if df.empty: continue
        if cfg.get('special') == 'g-model':
            df = df.reset_index(drop=True)
            df['generation'] = (df.index // 20) + 1
            gen_col, start_time, df_filtered = 'generation', df[cfg['time_col']].min(), df.copy()
        else:
            gen_col = cfg.get('gen_col')
            target_phase = cfg.get('target_phase')
            if target_phase in df[cfg['phase_col']].values:
                phase_data = df[df[cfg['phase_col']] == target_phase]
                start_time, df_filtered = phase_data[cfg['time_col']].min(), phase_data.copy()
            else: continue
        df_filtered['norm_time'] = df_filtered[cfg['time_col']] - start_time
        df_12h = df_filtered[df_filtered['norm_time'] <= MAX_H * 3600]
        is_crash = df_12h['success'].astype(str) == 'False'
        crashes = df_12h[is_crash].copy()
        if not crashes.empty:
            crashes = crashes.sort_values('norm_time')
            unique_crashes = crashes.drop_duplicates(subset=[cfg['input_col']], keep='first')
            if gen_col in unique_crashes.columns:
                g_series = unique_crashes[gen_col].dropna()
                if label == "Random": g_series = g_series.apply(lambda x: max(1, x))
                g_list = g_series[g_series > 0].tolist()
                if g_list: data_map[label] = g_list
    return data_map

def plot_save_single(data_map, filename):
    valid_keys = [m for m in METHODS_ORDER if m in data_map and len(data_map[m]) > 0 and m != "Random"]
    valid_keys.reverse() 
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    
    if not valid_keys:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=20) 
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    plot_data = [data_map[k] for k in valid_keys]
    colors = [COLOR_MAP.get(k, '#333') for k in valid_keys]
    
    for i, (k, vals) in enumerate(zip(valid_keys, plot_data)):
        y_pos = i + 1
        y_jitter = np.random.normal(y_pos, 0.08, size=len(vals))
        ax.scatter(vals, y_jitter, alpha=0.25, color=COLOR_MAP.get(k), s=24, edgecolor='none', zorder=2, rasterized=True) 

    box = ax.boxplot(plot_data, vert=False, patch_artist=True,
                     labels=valid_keys, showmeans=True,
                     widths=0.55, showfliers=False, zorder=10, 
                     meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6, "zorder":12, "alpha": 1},
                     medianprops={"color": "black", "linewidth": 1.5, "zorder":12, "alpha": 1},
                     boxprops={"linewidth": 1.5, "edgecolor": "black", "zorder":10}, 
                     whiskerprops={"linewidth": 1.5, "color": "black", "zorder":10},
                     capprops={"linewidth": 1.5, "color": "black", "zorder":10})
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_rasterized(True) 

    ax.set_xlabel('Generation (Log Scale)', fontsize=24, labelpad=10) 
    ax.set_xscale('symlog', linthresh=1) 
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    ax.tick_params(axis='x', labelsize=20) 
    ax.tick_params(axis='y', labelsize=20) 

    ax.grid(True, axis='x', which="major", ls="-", color="#e0e0e0", alpha=0.8, zorder=0)
    ax.grid(True, axis='x', which="minor", ls=":", color="#e0e0e0", alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5) 
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f"[Success] Saved {filename}")

def main():
    # 保持大字号配置
    plt.rcParams.update({
        'font.family': 'serif', 
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'], 
        'font.size': 20,    
        'pdf.fonttype': 42, 
        'ps.fonttype': 42,
        'axes.unicode_minus': False 
    })
    
    mc_data = get_mc_data(base_dir='mc')
    bw_data = get_bw_data(base_dir='bw')
    carla_data = get_carla_data(base_dir='carla')
    
    plot_save_single(mc_data, "RQ3_MC.pdf")
    plot_save_single(bw_data, "RQ3_BW.pdf")
    plot_save_single(carla_data, "RQ3_CARLA.pdf")

if __name__ == "__main__":
    main()