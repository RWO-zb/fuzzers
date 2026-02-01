import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from upsetplot import from_contents, plot as upset_plot
from venn import venn

OUT_UPSET = "Bipedal_Crash_Intersection_UpSet_NoRandG.pdf"
OUT_VENN = "Bipedal_Crash_Intersection_Venn_NoRandG.pdf"


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

FILES_CONFIG = {
    "CureFuzz": {
        "path": "selection_log.pkl",          
        "type": "pickle_curefuzz",
        "label": "CureFuzz"
    },
    "MDPFuzz": {
        "path": "fuzzer_10_0.01_0.01_0_logs.txt", 
        "type": "csv_mdpfuzz",
        "label": "MDPFuzz"
    },
    "QDFuzz": {
        "path": "1768120702.3916006_data.csv",            
        "type": "csv_qdfuzz",
        "label": "QDFuzz"
    },
    "SeqFuzz": {
        "path": "all_run_seeds_0.pkl",        
        "type": "pickle_seqfuzz",
        "label": "SeqFuzz"
    }
}

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def normalize_key(key_raw):
    
    if isinstance(key_raw, bytes):
        return str(key_raw)
    if isinstance(key_raw, (list, tuple, np.ndarray)):
        return str(tuple(key_raw))
    return str(key_raw)

class CrashSeedAnalyzer:
    def __init__(self):
        self.crash_sets = {}

    def load_data(self):
        print("Loading crash data...")
        
        for name, config in FILES_CONFIG.items():
            path = config['path']
            file_type = config['type']
            
            if not os.path.exists(path):
                print(f"  [Warn] {name}: File not found ({path})")
                continue
            
            unique_crashes = set()
            
            try:            
                if file_type == 'pickle_curefuzz':
                    data = load_pickle(path)
                    for entry in data:
                        if entry.get('did_crash', False):
                            state = entry.get('mutate_state')
                            raw_key = state.tobytes() if hasattr(state, 'tobytes') else str(state)
                            unique_crashes.add(normalize_key(raw_key))
                elif file_type == 'pickle_seqfuzz':
                    data = load_pickle(path)
                    for entry in data:
                        if entry.get('crashed', False):
                            state = entry.get('state')
                            raw_key = state.tobytes() if hasattr(state, 'tobytes') else str(state)
                            unique_crashes.add(normalize_key(raw_key))

                elif file_type == 'csv_mdpfuzz':
                    df = pd.read_csv(path, delimiter=';', engine='python', on_bad_lines='skip', skipinitialspace=True)
                    
                    if 'Oracle' in df.columns:
                        df['Oracle'] = df['Oracle'].astype(str).map({'True': True, 'False': False, 'None': False}).fillna(False)
                    
                    for col in ['BD_Distance', 'BD_MeanAngle']:
                        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.dropna(subset=['BD_Distance', 'BD_MeanAngle'], inplace=True)
                    
                    if 'rt_' not in os.path.basename(path).lower():
                        gen_col = next((c for c in df.columns if c.lower() == 'generation'), None)
                        if gen_col: 
                            df = df[(df[gen_col] != 0) & (df[gen_col].notna())]

                    crashes = df[df['Oracle'] == True]
                    for row in crashes.itertuples(index=False):
                        inp = getattr(row, 'Input', None)
                        if inp:
                            unique_crashes.add(normalize_key(inp))

                elif file_type == 'csv_qdfuzz':
                    df = pd.read_csv(path)
                    if 'is_faulty' in df.columns:
                        crashes = df[df['is_faulty'] == True]
                        for row in crashes.itertuples(index=False):
                            inp = getattr(row, 'input', None)
                            if inp:
                                unique_crashes.add(normalize_key(inp))

                self.crash_sets[name] = unique_crashes
                print(f"  {name}: Found {len(unique_crashes)} unique crashes.")

            except Exception as e:
                print(f"  [Error] {name}: {e}")
    
    def get_sets(self):
        return self.crash_sets

def main():
    analyzer = CrashSeedAnalyzer()
    analyzer.load_data()
    crash_sets = analyzer.get_sets()

    crash_sets = {k: v for k, v in crash_sets.items() if len(v) > 0}

    if not crash_sets:
        print("No crash data available for plotting.")
        return
    
    print("\nStarting plotting...")

    print(f"  Generating UpSet Plot -> {OUT_UPSET}")
    try:
        upset_data = from_contents(crash_sets)
        fig = plt.figure(figsize=(10, 6))
        upset_plot(upset_data, subset_size='count', show_counts=True, sort_by='cardinality', fig=fig)
        plt.title(f"Intersection of Crash Inputs (Fuzzers Only)", fontsize=16)
        plt.savefig(OUT_UPSET, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"  [Error] UpSet Plot: {e}")

    print(f"  Generating Venn Diagram -> {OUT_VENN}")
    try:
        plt.figure(figsize=(9, 9))
        venn(crash_sets, cmap="plasma", alpha=0.3, legend_loc="upper right")
        plt.title(f"Venn Diagram of Crash Inputs (Fuzzers Only)", fontsize=16)
        plt.savefig(OUT_VENN, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"  [Error] Venn Plot: {e}")

    print("\nDone. Please check the generated PDF files.")

if __name__ == "__main__":
    main()