import os
import pickle
import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

CONFIG = {
    'SeqFuzz': {
        'path': 'all_run_seeds_0.pkl',
        'enabled': True,
        'color': '#9C27B0',  
        'format': 'seqfuzz'
    },
    'MdpFuzz': {
        'path': 'MC_DQN_NoCov_5_0.01_0.1_0_12h_logs.txt', 
        'enabled': True,
        'color': '#E64A19', 
        'format': 'mdpfuzz'
    },
   
    'Random': {
        'path': 'MC_DQN_RT_0_budget730000_logs.txt',  
        'enabled': True,
        'color': '#757575',  
        'format': 'mdpfuzz' 
    },
    'CureFuzz': {
        'path': 'selection_log.pkl',
        'enabled': True,
        'color': '#2196F3',
        'format': 'curefuzz'
    },
    'QDFuzz': {
        'path': 'mc_test_data.csv',
        'enabled': True,
        'color': '#FFC107', 
        'format': 'qdfuzz'
    },
    'G-Model': {
        'path': 'all_test_cases_log.pkl',
        'enabled': True,
        'color': '#009688',  
        'format': 'gmodel'
    }
}

OUTPUT_FILENAME = 'compare_crash_discovery_over_time.png'
sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)


def load_pickle(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

def process_seqfuzz(filepath):
   
    data = load_pickle(filepath)
    if not data: return []
    
    seen_states = set()
    crash_times = []
    
    for entry in data:
        state = entry.get('state')
        if state is None: state = entry.get('mutate_state')
        if state is None: continue
        
        try:
            state_bytes = state.tobytes() if hasattr(state, 'tobytes') else bytes(state)
        except:
            continue
            
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            if entry.get('crashed') or entry.get('did_crash', False):
                t = entry.get('crash_time')
                if t is not None:
                    crash_times.append(t)
    
    return sorted(crash_times)

def process_mdpfuzz_style(filepath):
    if not os.path.exists(filepath):
        return []
    
    try:
       
        df = pd.read_csv(filepath, sep=';', skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        
      
        SKIP_INITIAL = 10000
        if len(df) > SKIP_INITIAL:
            df = df.iloc[SKIP_INITIAL:].copy()
            
        df['Oracle'] = df['Oracle'].astype(str).str.strip() == 'True'
        
       
        crashes = df[df['Oracle'] == True].copy()
        if crashes.empty: return []
      
        if 'CrashTime' not in crashes.columns:
            crashes['CrashTime'] = np.nan
        
        if 'RunTime' in crashes.columns:
            crashes['RunTime'] = pd.to_numeric(crashes['RunTime'], errors='coerce')
            start_time = df['RunTime'].min()
            
            mask = crashes['CrashTime'].isna()
            crashes.loc[mask, 'CrashTime'] = crashes.loc[mask, 'RunTime'] - start_time
            
        unique_crashes = crashes.drop_duplicates(subset=['Input'])
        
        crash_times = unique_crashes['CrashTime'].dropna().values.tolist()
        return sorted(crash_times)
        
    except Exception as e:
        return []

def process_curefuzz(filepath):
    
    data = load_pickle(filepath)
    if not data: return []
    
    seen_states = set()
    crash_times = []
    
    for entry in data:
        state = entry.get('mutate_state')
        if state is None: continue
        
        state_bytes = state.tobytes()
        
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            if entry.get('did_crash', False):
                t = entry.get('crash_time')
                if t is not None:
                    crash_times.append(t)
                    
    return sorted(crash_times)

def process_qdfuzz(filepath):
   
    if not os.path.exists(filepath):
        return []
    
    try:
        df = pd.read_csv(filepath)
        crash_times = []
        seen_states = set()
        
        for _, row in df.iterrows():
         
            if not row.get('is_faulty', False):
                continue
                
            inp = row.get('input')
            if isinstance(inp, str):
                try:
                    inp = ast.literal_eval(inp)
                except:
                    pass
            
            state = np.array(inp, dtype=np.float32) if isinstance(inp, list) else inp
            if state is None: continue
            
            try:
                state_bytes = state.tobytes()
                if state_bytes not in seen_states:
                    seen_states.add(state_bytes)
                    t = row.get('discovery_time')
                    if t is not None:
                        crash_times.append(t)
            except:
                continue
                
        return sorted(crash_times)
    except Exception as e:
        return []

def process_gmodel(filepath):
    data = load_pickle(filepath)
    if not data: return []
    
    seen_inputs = set()
    crash_times = []
    
    for entry in data:
      
        if entry.get('is_crash'):
            t_in = tuple(entry['input'])
            if t_in not in seen_inputs:
                seen_inputs.add(t_in)
                t = entry.get('timestamp', 0)
                crash_times.append(t)
                
    return sorted(crash_times)


def plot_combined_crashes():
    plt.figure(figsize=(12, 8))
    
    max_time_hours = 0
    has_data = False
    
    for label, config in CONFIG.items():
        if not config['enabled']:
            continue
            
        fmt = config['format']
        path = config['path']
        
        times = []
        if fmt == 'seqfuzz':
            times = process_seqfuzz(path)
        elif fmt == 'mdpfuzz':
            times = process_mdpfuzz_style(path)
        elif fmt == 'curefuzz':
            times = process_curefuzz(path)
        elif fmt == 'qdfuzz':
            times = process_qdfuzz(path)
        elif fmt == 'gmodel':
            times = process_gmodel(path)
            
        if not times:
            continue
            
        has_data = True
        
      
        times_hours = [t / 3600.0 for t in times]
        counts = list(range(1, len(times) + 1))
        
        if times_hours:
            max_time_hours = max(max_time_hours, max(times_hours))
        
        plt.plot(times_hours, counts, 
                 label=label, 
                 color=config['color'], 
                 linewidth=2.5, 
                 alpha=0.85)
        
        if times_hours:
            plt.scatter(times_hours[-1], counts[-1], color=config['color'], s=40)

    if not has_data:
        return

 
    plt.title('Unique Crash Discovery Over Time (Method Comparison)', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Time (Hours)', fontsize=14, labelpad=10)
    plt.ylabel('Cumulative Unique Crashes', fontsize=14, labelpad=10)
    
    plt.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    sns.despine()
    plt.tight_layout()
    
    
    plt.savefig(OUTPUT_FILENAME, dpi=300)
   
    plt.close()

if __name__ == "__main__":
    plot_combined_crashes()