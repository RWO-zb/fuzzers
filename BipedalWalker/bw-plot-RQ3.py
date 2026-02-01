import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pickle
import os
import math

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

COLOR_MAP = {
    "CureFuzz": "#1f77b4", # Blue
    "G-Model":  "#ff7f0e", # Orange
    "MDPFuzz":  "#2ca02c", # Green
    "QDFuzz":   "#d62728", # Red
    "Random":   "#9467bd", # Purple
    "SeqFuzz":  "#8c564b", # Brown
}


MAX_HOURS = 12.0

G_MODEL_STEP_SIZE = 50 

FILES_CONFIG = {
    "CureFuzz": { "path": "selection_log.pkl", "type": "cure_pkl" },
    "G-Model":  { "path": "all_test_cases_log.pkl", "type": "gmodel_pkl" },
    "MDPFuzz":  { "path": "fuzzer_10_0.01_0.01_0_logs.txt", "type": "mdpfuzz_txt" },
    "QDFuzz":   { "path": "1765639810.5339673_data.csv", "type": "qdfuzz_csv" },
    "Random":   { "path": "rt_10_0.01_0.01_0_logs.txt", "type": "mdpfuzz_txt" },
    "SeqFuzz":  { "path": "all_run_seeds_0.pkl", "type": "seq_pkl" }
}


def load_mdpfuzz_txt(path):
    
    try:
        df = pd.read_csv(path, delimiter=';', on_bad_lines='skip', skipinitialspace=True)
        
        if 'Oracle' in df.columns:
            if df['Oracle'].dtype == 'object':
                df['Oracle'] = df['Oracle'].map({'True': True, 'False': False, 'None': None})
            df['is_crash'] = (df['Oracle'] == True)
        elif 'success' in df.columns:
            df['is_crash'] = (df['success'] == False)
        
       
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
            
            
            if isinstance(inp, list):
                inp_bytes = tuple(inp)
            elif isinstance(inp, np.ndarray):
                inp_bytes = tuple(inp.tolist())
            else:
                inp_bytes = inp 
            
            
            step = entry.get('step', 0)
            
            records.append({
                'time': entry.get('time'),
                'input': inp_bytes,
                'is_crash': entry.get('is_crash', False),
                'step': step,
                'generation': np.nan 
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[Loader Error] G-Model ({path}): {e}")
        return None

def load_qdfuzz_csv(path):
  
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

LOADERS = {
    "mdpfuzz_txt": load_mdpfuzz_txt,
    "cure_pkl": load_cure_pkl,
    "seq_pkl": load_seq_pkl,
    "gmodel_pkl": load_gmodel_pkl,
    "qdfuzz_csv": load_qdfuzz_csv
}


def process_data(label, config):
    path = config["path"]
    loader_type = config["type"]
    
    if not os.path.exists(path):
        
        return None, None, []

    
    loader = LOADERS.get(loader_type)
    df = loader(path)
    
    if df is None or df.empty:
       
        return None, None, []

    
    if 'time' not in df.columns or df['time'].isnull().all():
        return None, None, []
        
    start_time = df['time'].min()
    df['norm_time'] = df['time'] - start_time
    
    limit_sec = MAX_HOURS * 3600
    df_period = df[df['norm_time'] <= limit_sec].copy()
    
  
    crashes = df_period[df_period['is_crash'] == True].copy()
    
    if crashes.empty:
        return 0, np.nan, [] 

   
    crashes = crashes.sort_values('norm_time')
    unique_crashes = crashes.drop_duplicates(subset=['input'], keep='first').copy()
    
    
    if label == "G-Model" and 'step' in unique_crashes.columns:
    
        unique_crashes['generation'] = unique_crashes['step'].apply(
            lambda x: math.ceil(x / G_MODEL_STEP_SIZE)
        )
    
    
    if 'generation' in unique_crashes.columns:
        n_before = len(unique_crashes)
        unique_crashes = unique_crashes[unique_crashes['generation'] > 0]
        n_after = len(unique_crashes)
        if n_before != n_after:
            print(f"  [{label}] Filtered out {n_before - n_after} initial seeds (Gen 0).")

   
    n_crashes = len(unique_crashes)
    

    if n_crashes > 0:
        time_eff = (MAX_HOURS * 60) / n_crashes
    else:
        time_eff = 0
        
    gen_list = []
    if 'generation' in unique_crashes.columns:
        gen_series = unique_crashes['generation'].dropna()
        if not gen_series.empty:
            gen_list = gen_series.tolist()

    avg_gen = np.nan
    if gen_list:
         avg_gen = np.mean(gen_list)
            
    print(f"  -> {label}: {n_crashes} valid crashes, Cost={time_eff:.1f} min, AvgGen={avg_gen:.1f}")
    return time_eff, avg_gen, gen_list



def plot_bar_charts(bar_metrics):
   
    if not bar_metrics["labels"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    bars1 = ax1.bar(bar_metrics["labels"], bar_metrics["time_per_crash"], 
                    color=bar_metrics["colors"], alpha=0.8, edgecolor='black', width=0.6)
    
    ax1.set_title("Time Cost Efficiency", fontweight='bold')
    ax1.set_ylabel("Avg. Minutes per Crash")
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
 
    for bar in bars1:
        height = bar.get_height()
        label_text = f'{height:.1f}' if height > 0 else 'N/A'
        ax1.text(bar.get_x() + bar.get_width()/2., height, label_text,
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2 = axes[1]
    
    valid_indices = [i for i, x in enumerate(bar_metrics["gen_avg_depth"]) if not np.isnan(x) and x > 0]
    
    if valid_indices:
        v_labels = [bar_metrics["labels"][i] for i in valid_indices]
        v_values = [bar_metrics["gen_avg_depth"][i] for i in valid_indices]
        v_colors = [bar_metrics["colors"][i] for i in valid_indices]

        bars2 = ax2.bar(v_labels, v_values, color=v_colors, 
                        alpha=0.8, edgecolor='black', width=0.6)
        
        ax2.set_title("Avg. Generation of Discovery", fontweight='bold')
        ax2.set_ylabel("Avg. Generation Index")
        ax2.grid(axis='y', linestyle='--', alpha=0.6)

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
    output_file = 'RQ3_bar_charts.png'
    plt.savefig(output_file, dpi=300)

def plot_generation_distribution(gen_data_map):
    valid_data = {k: v for k, v in gen_data_map.items() if v and len(v) > 0}
    
    if not valid_data:
        return

    labels = list(valid_data.keys())
    data_values = list(valid_data.values())
    colors = [COLOR_MAP.get(lbl, '#333') for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    box = ax.boxplot(data_values, vert=False, patch_artist=True,
                     labels=labels, showmeans=True,
                     widths=0.6,
                     meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":8},
                     medianprops={"color": "black", "linewidth": 1.5})

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    for i, (method, values) in enumerate(valid_data.items()):
        y_pos = i + 1
        y_jitter = np.random.normal(y_pos, 0.08, size=len(values))
        ax.scatter(values, y_jitter, alpha=0.6, color=COLOR_MAP.get(method), 
                   s=15, edgecolor='white', linewidth=0.5)

    ax.set_title('Distribution of Unique Crashes by Generation (Mutants Only)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Generation Number (Log Scale)', fontsize=14)
    
    ax.set_xscale('symlog', linthresh=1)
    
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    
    filename = 'RQ3_generation_boxplot_log.png'
    plt.savefig(filename, dpi=300)
    
def main():
    bar_metrics = {
        "labels": [],
        "time_per_crash": [],
        "gen_avg_depth": [],
        "colors": []
    }
    
    boxplot_data = {}
    
    for label, config in FILES_CONFIG.items():
        time_eff, avg_gen, gen_list = process_data(label, config)
        
        if time_eff is not None:
            bar_metrics["labels"].append(label)
            bar_metrics["time_per_crash"].append(time_eff)
            bar_metrics["gen_avg_depth"].append(avg_gen)
            bar_metrics["colors"].append(COLOR_MAP.get(label, "#333333"))
        
        if label != "Random" and gen_list:
            boxplot_data[label] = gen_list

    
    plot_bar_charts(bar_metrics)
    
    plot_generation_distribution(boxplot_data)
    

if __name__ == "__main__":
    main()