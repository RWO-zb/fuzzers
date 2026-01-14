import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import json

OUTPUT_FILE = 'RQ2_BipedalWalker.png'
GRID_SIZE = (50, 50) 

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

FILES_CONFIG = {
    "CureFuzz": {
        "path": "selection_log.pkl",          
        "type": "pickle_curefuzz",
        "color": "C0",                        
        "label": "CureFuzz"
    },
    "G-Model": {
        "path": "all_test_cases_log.pkl",     
        "type": "pickle_gmodel",
        "color": "C1",                        
        "label": "G-Model"
    },
    "MDPFuzz": {
        "path": "fuzzer_10_0.01_0.01_0_logs.txt", 
        "type": "csv_mdpfuzz",
        "color": "C2",                        
        "label": "MDPFuzz"
    },
    "QDFuzz": {
        "path": "1768120702.3916006_data.csv",            
        "type": "csv_qdfuzz",
        "color": "C3",                        
        "label": "QDFuzz"
    },
    "Random": {
        "path": "rt_10_0.01_0.01_1_logs.txt", 
        "type": "csv_mdpfuzz",                
        "color": "C4",                        
        "label": "Random"
    },
    "SeqFuzz": {
        "path": "all_run_seeds_0.pkl",        
        "type": "pickle_seqfuzz",
        "color": "C5",                        
        "label": "SeqFuzz"
    }
}

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def load_data_standardized(method_name, config):
    path = config['path']
    file_type = config['type']
    standardized_data = []

    print(f"Loading {method_name} from {path}...")
    
    # --- CureFuzz ---
    if file_type == 'pickle_curefuzz':
        data = load_pickle(path)
        for entry in data:
            d = entry.get('bd_distance')
            a = entry.get('bd_mean_angle')
            state = entry.get('mutate_state')
            crash = entry.get('did_crash', False)
            if d is not None and a is not None:
                state_key = state.tobytes() if hasattr(state, 'tobytes') else str(state)
                standardized_data.append({'bd_0': d, 'bd_1': a, 'is_crash': crash, 'state_key': state_key})

    # --- G-Model ---
    elif file_type == 'pickle_gmodel':
        data = load_pickle(path)
        for entry in data:
            d = entry.get('bd_distance')
            a = entry.get('bd_mean_angle')
            inp = entry.get('input')
            crash = entry.get('is_crash', False)
            if d is not None and a is not None:
                state_key = tuple(inp) if isinstance(inp, list) else (inp.tobytes() if hasattr(inp, 'tobytes') else str(inp))
                standardized_data.append({'bd_0': d, 'bd_1': a, 'is_crash': crash, 'state_key': state_key})

    # --- SeqFuzz ---
    elif file_type == 'pickle_seqfuzz':
        data = load_pickle(path)
        for entry in data:
            d = entry.get('bd_distance')
            a = entry.get('bd_mean_angle')
            state = entry.get('state')
            crash = entry.get('crashed', False)
            if d is not None and a is not None:
                state_key = state.tobytes() if hasattr(state, 'tobytes') else str(state)
                standardized_data.append({'bd_0': d, 'bd_1': a, 'is_crash': crash, 'state_key': state_key})

    # --- MDPFuzz / Random ---
    elif file_type == 'csv_mdpfuzz':
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            return []
       
        df = pd.read_csv(path, delimiter=';', engine='python', on_bad_lines='skip', skipinitialspace=True)
        if 'Oracle' in df.columns:
            df['Oracle'] = df['Oracle'].astype(str).map({'True': True, 'False': False, 'None': False}).fillna(False)
            
            # 数值转换
        for col in ['BD_Distance', 'BD_MeanAngle']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['BD_Distance', 'BD_MeanAngle'], inplace=True)
            
            # 过滤 MDPFuzz 的 Generation 0 (保留 Random 的)
        if 'rt_' not in os.path.basename(path).lower():
            gen_col = next((c for c in df.columns if c.lower() == 'generation'), None)
            if gen_col: df = df[(df[gen_col] != 0) & (df[gen_col].notna())]

        for row in df.itertuples(index=False):
            d = getattr(row, 'BD_Distance', None)
            a = getattr(row, 'BD_MeanAngle', None)
            inp = getattr(row, 'Input', None)
            crash = getattr(row, 'Oracle', False)
            if d is not None:
                standardized_data.append({'bd_0': d, 'bd_1': a, 'is_crash': crash, 'state_key': str(inp)})
        
    # --- QDFuzz ---
    elif file_type == 'csv_qdfuzz':
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            return []
       
        df = pd.read_csv(path)
            
        if 'elapsed_time' in df.columns:
            df = df.sort_values('elapsed_time')
            
        for row in df.itertuples(index=False):
            d = getattr(row, 'behavior0', None)
            a = getattr(row, 'behavior1', None)
            inp = getattr(row, 'input', None)
            crash = getattr(row, 'is_faulty', False)
            if d is not None:
                standardized_data.append({'bd_0': d, 'bd_1': a, 'is_crash': crash, 'state_key': str(inp)})
       
    print(f"  -> Loaded {len(standardized_data)} entries.")
    return standardized_data

def get_global_ranges(all_datasets):
    all_b0, all_b1 = [], []
    for _, data in all_datasets.items():
        for entry in data:
            all_b0.append(entry['bd_0'])
            all_b1.append(entry['bd_1'])
    
    if not all_b0: return (0, 1), (0, 1)
    
    min_b0, max_b0 = min(all_b0), max(all_b0) + 1e-5
    min_b1, max_b1 = min(all_b1), max(all_b1) + 1e-5
    
    print(f"\n[Global Range] Distance: [{min_b0:.2f}, {max_b0:.2f}], Angle: [{min_b1:.2f}, {max_b1:.2f}]")
    return (min_b0, max_b0), (min_b1, max_b1)

def get_bin_index(value, min_val, max_val, grid_dim):
    if max_val <= min_val: return 0
    idx = int((value - min_val) / (max_val - min_val) * grid_dim)
    return min(max(idx, 0), grid_dim - 1)

def compute_metrics(data, ranges):
    (min_b0, max_b0), (min_b1, max_b1) = ranges
    bd_filled = set()
    fd_filled = set()
    unique_states = set()
    
    results = {'bd': [], 'fd': [], 'sc': []}
    
    for entry in data:
        if entry['state_key'] is not None:
            unique_states.add(entry['state_key'])
        results['sc'].append(len(unique_states))
        
        idx0 = get_bin_index(entry['bd_0'], min_b0, max_b0, GRID_SIZE[0])
        idx1 = get_bin_index(entry['bd_1'], min_b1, max_b1, GRID_SIZE[1])
        bin_loc = (idx0, idx1)
        
        bd_filled.add(bin_loc)
        results['bd'].append(len(bd_filled))
        
        if entry['is_crash']:
            fd_filled.add(bin_loc)
        results['fd'].append(len(fd_filled))
        
    return results


def main():
    all_datasets = {}
    for name, config in FILES_CONFIG.items():
        all_datasets[name] = load_data_standardized(name, config)
        
    ranges = get_global_ranges(all_datasets)
    
    all_trends = {}
    for name, data in all_datasets.items():
        if data:
            all_trends[name] = compute_metrics(data, ranges)
            
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_info = [
        ('sc', 'State Coverage Growth', 'Cumulative Unique Inputs'),
        ('bd', 'Behaviour Diversity Growth', 'Cumulative Covered Bins'),
        ('fd', 'Fault Diversity Growth', 'Cumulative Crash Bins')
    ]
    
    for i, (key, title, ylabel) in enumerate(metrics_info):
        ax = axes[i]
        
        for name, config in FILES_CONFIG.items():
            if name not in all_trends: continue
            
            trend = all_trends[name][key]
            if not trend: continue
            
            x = range(1, len(trend) + 1)
            ax.plot(x, trend, label=config['label'], color=config['color'], alpha=0.9)
            
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Number of Test Cases')
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"\n[Success] Plot saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()