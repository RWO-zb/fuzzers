import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    'legend.fontsize': 12,
    'lines.linewidth': 2
})


COLORS = {
    'Random':   '#7f8c8d',  
    'MDPFuzz':  '#e74c3c',  
    'SeqFuzz':  '#2ecc71',  
    'CureFuzz': '#9b59b6',  
    'QDFuzz':   '#f39c12',  
    'G-Model':  '#3498db',  
}

CONFIG = [
    {
        'label': 'Random',
        'file_path': 'rt_10_0.01_0.01_0_logs.txt',  
        'parser': 'mdpfuzz',                
        'color': COLORS['Random']
    },
    {
        'label': 'MDPFuzz',
        'file_path': 'fuzzer_10_0.01_0.01_0_logs.txt', 
        'parser': 'mdpfuzz',
        'color': COLORS['MDPFuzz']
    },
    {
        'label': 'SeqFuzz',
        'file_path': 'all_run_seeds_0.pkl', 
        'parser': 'seqfuzz',
        'color': COLORS['SeqFuzz']
    },
    {
        'label': 'CureFuzz',
        'file_path': 'selection_log.pkl',   
        'parser': 'curefuzz',
        'color': COLORS['CureFuzz']
    },
    {
        'label': 'QDFuzz',
        'file_path': '1765639810.5339673_data.csv',     
        'parser': 'qdfuzz',
        'color': COLORS['QDFuzz']
    },
    {
        'label': 'G-Model',
        'file_path': 'all_test_cases_log.pkl', 
        'parser': 'gmodel',
        'color': COLORS['G-Model']
    }
]


max_h = 12.0        
view_limit_h = 12.5 



def parse_mdpfuzz_format(file_path):
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return np.array([])
    try:
        df = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip', skipinitialspace=True)
        if 'Oracle' in df.columns and df['Oracle'].dtype == 'object':
            df['Oracle'] = df['Oracle'].map({'True': True, 'False': False, 'None': None})
        
        df['is_crash'] = (df['Oracle'] == True)
        
        if 'RunTime' in df.columns:
            df['RunTime'] = pd.to_numeric(df['RunTime'], errors='coerce')
        else:
            return np.array([])

        if 'Input' not in df.columns:
            return np.array([])
            
        unique_df = df.drop_duplicates(subset=['Input'], keep='first')
        crash_df = unique_df[unique_df['is_crash'] == True].copy()
        
        if crash_df.empty:
            return np.array([])

        start_time = df['RunTime'].min()
        crash_times = crash_df['RunTime'] - start_time
        return np.sort(crash_times.values)

    except Exception as e:
        print(f"Error parsing MDPFuzz {file_path}: {e}")
        return np.array([])


def parse_seqfuzz_format(file_path):
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return np.array([])
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        seen_states = set()
        dedup_crashes_timestamps = []
        int32_size = 60
        int64_size = 120
        
        for entry in data:
            state = entry.get('state')
            if state is None: continue
            try:
                state_bytes = state.tobytes()
            except AttributeError:
                continue
            if len(state_bytes) not in [int32_size, int64_size]:
                continue
                
            if state_bytes not in seen_states:
                seen_states.add(state_bytes)
                if entry.get('crashed', False):
                    t = entry.get('timestamp')
                    if t is not None:
                        dedup_crashes_timestamps.append(t)
        
        dedup_crashes_timestamps.sort()
        if dedup_crashes_timestamps and dedup_crashes_timestamps[0] > 1e9:
             start_time = min(dedup_crashes_timestamps)
             dedup_crashes_timestamps = [t - start_time for t in dedup_crashes_timestamps]

        return np.array(dedup_crashes_timestamps)
    except Exception as e:
        print(f"Error parsing SeqFuzz {file_path}: {e}")
        return np.array([])


def parse_curefuzz_format(file_path):
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return np.array([])
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        seen_states = set()
        crash_times = []
        int32_size = 60
        int64_size = 120

        for entry in data:
            state = entry.get('mutate_state')
            if state is None: continue
            try:
                state_bytes = state.tobytes()
            except AttributeError:
                continue
            if len(state_bytes) not in [int32_size, int64_size]:
                continue
            
            if state_bytes not in seen_states:
                seen_states.add(state_bytes)
                if entry.get('did_crash', False):
                    t = entry.get('elapsed_time')
                    if t is not None:
                        crash_times.append(t)
        
        crash_times.sort()
        return np.array(crash_times)
    except Exception as e:
        print(f"Error parsing CureFuzz {file_path}: {e}")
        return np.array([])


def parse_qdfuzz_format(file_path):
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return np.array([])
    try:
        df = pd.read_csv(file_path)
        if 'input' not in df.columns or 'is_faulty' not in df.columns or 'elapsed_time' not in df.columns:
            return np.array([])
            
        if df['is_faulty'].dtype == 'object':
             df['is_faulty'] = df['is_faulty'].astype(str).str.lower() == 'true'
        else:
             df['is_faulty'] = df['is_faulty'].astype(bool)
             
        df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce').fillna(0.0)
        unique_df = df.dropna(subset=['input']).drop_duplicates(subset=['input'], keep='first')
        crash_df = unique_df[unique_df['is_faulty'] == True]
        
        crash_times = sorted(crash_df['elapsed_time'].tolist())
        return np.array(crash_times)
    except Exception as e:
        print(f"Error parsing QDFuzz {file_path}: {e}")
        return np.array([])


def parse_gmodel_format(file_path):
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return np.array([])
    try:
        with open(file_path, 'rb') as f:
            log_data = pickle.load(f)

        unique_crashes_timestamps = {}
        for entry in log_data:
            if entry.get('is_crash'):
                t_in = tuple(entry['input'])
                timestamp = entry.get('time', 0)
                if t_in not in unique_crashes_timestamps:
                    unique_crashes_timestamps[t_in] = timestamp
        
        crash_times = sorted(list(unique_crashes_timestamps.values()))
        return np.array(crash_times)
    except Exception as e:
        print(f"Error parsing G-Model {file_path}: {e}")
        return np.array([])


plt.figure(figsize=(10, 6))

markers_x_h = np.arange(2, max_h + 0.1, 2)

for cfg in CONFIG:
    label = cfg['label']
    fpath = cfg['file_path']
    parser = cfg['parser']
    color = cfg['color']
    
    print(f"Processing {label}...")
    
    times = np.array([])
    if parser == 'mdpfuzz':
        times = parse_mdpfuzz_format(fpath)
    elif parser == 'seqfuzz':
        times = parse_seqfuzz_format(fpath)
    elif parser == 'curefuzz':
        times = parse_curefuzz_format(fpath)
    elif parser == 'qdfuzz':
        times = parse_qdfuzz_format(fpath)
    elif parser == 'gmodel':
        times = parse_gmodel_format(fpath)
        
    if len(times) == 0:
        times_h = np.array([])
    else:

        limit_sec = max_h * 3600
        times = times[times <= limit_sec]
        times_h = times / 3600.0

    if len(times_h) > 0:
        x_plot = np.concatenate(([0], times_h))
        y_plot = np.concatenate(([0], np.arange(1, len(times_h) + 1)))
    else:
        x_plot = np.array([0])
        y_plot = np.array([0])
    
   
    line, = plt.step(x_plot, y_plot, where='post', label=label, color=color)
    
  
    
    actual_max_time = x_plot[-1] if len(x_plot) > 0 else 0
    
    marker_x_to_plot = []
    marker_y_to_plot = []
    
    for mx in markers_x_h:
        if mx > actual_max_time:
            continue
            
        if len(times_h) > 0:
            count = np.searchsorted(times_h, mx, side='right')
        else:
            count = 0
        
        marker_x_to_plot.append(mx)
        marker_y_to_plot.append(count)
        
    if marker_x_to_plot:
        plt.plot(marker_x_to_plot, marker_y_to_plot, linestyle='none', marker='^', 
                 color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)

plt.xlim(0, view_limit_h)
plt.xticks(np.arange(0, 13, 2))
plt.xlabel("Time (h)")
plt.ylabel("Number of Unique Crashes")
plt.title("Cumulative Unique Crashes (Comparison)")
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('RQ1-Comparison_Fixed.png', dpi=300)
print("Done. Saved to RQ1-Comparison_Fixed.png")