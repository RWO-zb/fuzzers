import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

try:
    from venn import venn
except ModuleNotFoundError:
    venn = None

try:
    from upsetplot import from_contents, plot as upset_plot
except ModuleNotFoundError:
    from_contents = None
    upset_plot = None

BASE_DIR = "seed0"
OUT_UPSET = os.path.join(BASE_DIR, "Bipedal_Crash_Intersection_UpSet_Seed0_AllMethods.pdf")
OUT_VENN = os.path.join(BASE_DIR, "Bipedal_Crash_Intersection_Venn_Seed0_AllMethods.pdf")
OUT_FD_UPSET = os.path.join(BASE_DIR, "Bipedal_Fault_Diversity_UpSet_Seed0_AllMethods.pdf")
OUT_FD_VENN = os.path.join(BASE_DIR, "Bipedal_Fault_Diversity_Venn_Seed0_AllMethods.pdf")
GRID_SIZE = (50, 50)


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

FILES_CONFIG = {
    "CureFuzz": {
        "path": os.path.join(BASE_DIR, "selection_log.pkl"),
        "type": "pickle_curefuzz",
        "label": "CureFuzz"
    },
    "G-Model": {
        "path": os.path.join(BASE_DIR, "all_test_cases_log.pkl"),
        "type": "pickle_gmodel",
        "label": "G-Model"
    },
    "MDPFuzz": {
        "path": os.path.join(BASE_DIR, "fuzzer_10_0.01_0.01_0_logs.txt"),
        "type": "csv_mdpfuzz",
        "label": "MDPFuzz"
    },
    "QDFuzz": {
        "path": os.path.join(BASE_DIR, "1769257333.763425_data.csv"),
        "type": "csv_qdfuzz",
        "label": "QDFuzz"
    },
    "Random": {
        "path": os.path.join(BASE_DIR, "rt_10_0.01_0.01_0_logs.txt"),
        "type": "csv_mdpfuzz",
        "label": "Random"
    },
    "SeqFuzz": {
        "path": os.path.join(BASE_DIR, "all_run_seeds_0.pkl"),
        "type": "pickle_seqfuzz",
        "label": "SeqFuzz"
    }
}

def load_pickle(path):
    if not hasattr(np, "_core"):
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric
        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
        sys.modules.setdefault("numpy._core.numeric", numpy_numeric)

    with open(path, 'rb') as f:
        return pickle.load(f)

def normalize_key(key_raw):
    
    if isinstance(key_raw, bytes):
        return str(key_raw)
    if isinstance(key_raw, (list, tuple, np.ndarray)):
        return str(tuple(key_raw))
    return str(key_raw)

def parse_bool(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)

def get_bin_index(value, min_val, max_val, grid_dim):
    if max_val <= min_val:
        return 0
    idx = int((value - min_val) / (max_val - min_val) * grid_dim)
    return min(max(idx, 0), grid_dim - 1)

def get_global_ranges(all_records):
    all_b0, all_b1 = [], []
    for records in all_records.values():
        for record in records:
            all_b0.append(record["bd_0"])
            all_b1.append(record["bd_1"])

    if not all_b0:
        return (0, 1), (0, 1)

    min_b0, max_b0 = min(all_b0), max(all_b0) + 1e-5
    min_b1, max_b1 = min(all_b1), max(all_b1) + 1e-5
    print(f"\n[Global FD Range] Distance: [{min_b0:.2f}, {max_b0:.2f}], Angle: [{min_b1:.2f}, {max_b1:.2f}]")
    return (min_b0, max_b0), (min_b1, max_b1)

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
                elif file_type == 'pickle_gmodel':
                    data = load_pickle(path)
                    for entry in data:
                        if parse_bool(entry.get('is_crash', False)):
                            inp = entry.get('input')
                            raw_key = tuple(inp) if isinstance(inp, list) else (inp.tobytes() if hasattr(inp, 'tobytes') else str(inp))
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

class FaultDiversityAnalyzer:
    def __init__(self):
        self.records = {}
        self.fault_bin_sets = {}

    def load_data(self):
        print("\nLoading behavior data for fault diversity...")

        for name, config in FILES_CONFIG.items():
            path = config['path']
            file_type = config['type']

            if not os.path.exists(path):
                print(f"  [Warn] {name}: File not found ({path})")
                continue

            records = []

            try:
                if file_type == 'pickle_curefuzz':
                    data = load_pickle(path)
                    for entry in data:
                        d = entry.get('bd_distance')
                        a = entry.get('bd_mean_angle')
                        if d is not None and a is not None:
                            records.append({
                                "bd_0": float(d),
                                "bd_1": float(a),
                                "is_crash": parse_bool(entry.get('did_crash', False))
                            })

                elif file_type == 'pickle_gmodel':
                    data = load_pickle(path)
                    for entry in data:
                        d = entry.get('bd_distance')
                        a = entry.get('bd_mean_angle')
                        if d is not None and a is not None:
                            records.append({
                                "bd_0": float(d),
                                "bd_1": float(a),
                                "is_crash": parse_bool(entry.get('is_crash', False))
                            })

                elif file_type == 'pickle_seqfuzz':
                    data = load_pickle(path)
                    for entry in data:
                        d = entry.get('bd_distance')
                        a = entry.get('bd_mean_angle')
                        if d is not None and a is not None:
                            records.append({
                                "bd_0": float(d),
                                "bd_1": float(a),
                                "is_crash": parse_bool(entry.get('crashed', False))
                            })

                elif file_type == 'csv_mdpfuzz':
                    df = pd.read_csv(path, delimiter=';', engine='python', on_bad_lines='skip', skipinitialspace=True)

                    if 'Oracle' in df.columns:
                        df['Oracle'] = df['Oracle'].astype(str).map({'True': True, 'False': False, 'None': False}).fillna(False)

                    for col in ['BD_Distance', 'BD_MeanAngle']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.dropna(subset=['BD_Distance', 'BD_MeanAngle'], inplace=True)

                    if 'rt_' not in os.path.basename(path).lower():
                        gen_col = next((c for c in df.columns if c.lower() == 'generation'), None)
                        if gen_col:
                            df = df[(df[gen_col] != 0) & (df[gen_col].notna())]

                    for row in df.itertuples(index=False):
                        records.append({
                            "bd_0": float(getattr(row, 'BD_Distance')),
                            "bd_1": float(getattr(row, 'BD_MeanAngle')),
                            "is_crash": parse_bool(getattr(row, 'Oracle', False))
                        })

                elif file_type == 'csv_qdfuzz':
                    df = pd.read_csv(path)
                    if 'elapsed_time' in df.columns:
                        df = df[pd.to_numeric(df['elapsed_time'], errors='coerce') > 0]
                    elif 'mutation_count' in df.columns:
                        df = df[pd.to_numeric(df['mutation_count'], errors='coerce') > 0]

                    for row in df.itertuples(index=False):
                        d = getattr(row, 'behavior0', None)
                        a = getattr(row, 'behavior1', None)
                        if d is not None and a is not None:
                            records.append({
                                "bd_0": float(d),
                                "bd_1": float(a),
                                "is_crash": parse_bool(getattr(row, 'is_faulty', False))
                            })

                self.records[name] = records
                print(f"  {name}: Loaded {len(records)} behavior records.")

            except Exception as e:
                print(f"  [Error] {name}: {e}")

    def compute_fault_bins(self):
        ranges = get_global_ranges(self.records)
        (min_b0, max_b0), (min_b1, max_b1) = ranges

        for name, records in self.records.items():
            fault_bins = set()
            for record in records:
                if not record["is_crash"]:
                    continue

                idx0 = get_bin_index(record["bd_0"], min_b0, max_b0, GRID_SIZE[0])
                idx1 = get_bin_index(record["bd_1"], min_b1, max_b1, GRID_SIZE[1])
                fault_bins.add(f"bin_{idx0}_{idx1}")

            self.fault_bin_sets[name] = fault_bins
            print(f"  {name}: Found {len(fault_bins)} fault-diverse bins.")

    def get_sets(self):
        return self.fault_bin_sets

def main():
    analyzer = CrashSeedAnalyzer()
    analyzer.load_data()
    crash_sets = analyzer.get_sets()

    crash_sets = {k: v for k, v in crash_sets.items() if len(v) > 0}

    if not crash_sets:
        print("No crash data available for plotting.")
        return
    
    print("\nStarting plotting...")

    if from_contents is None or upset_plot is None:
        print("  [Skip] UpSetPlot is not installed; skipping crash input UpSet plot.")
    else:
        print(f"  Generating UpSet Plot -> {OUT_UPSET}")
        try:
            upset_data = from_contents(crash_sets)
            fig = plt.figure(figsize=(11, 6))
            upset_plot(upset_data, subset_size='count', show_counts=True, sort_by='cardinality', fig=fig)
            plt.title(f"Intersection of Crash Inputs (All Methods)", fontsize=16)
            plt.savefig(OUT_UPSET, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"  [Error] UpSet Plot: {e}")

    print(f"  Generating Venn Diagram -> {OUT_VENN}")
    if venn is None:
        print("  [Skip] venn is not installed; skipping crash input Venn plot.")
    else:
        try:
            plt.figure(figsize=(11, 11))
            venn(crash_sets, cmap="plasma", alpha=0.3, legend_loc="upper right")
            plt.title(f"Venn Diagram of Crash Inputs (All Methods)", fontsize=16)
            plt.savefig(OUT_VENN, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"  [Error] Venn Plot: {e}")

    fd_analyzer = FaultDiversityAnalyzer()
    fd_analyzer.load_data()
    fd_analyzer.compute_fault_bins()
    fault_bin_sets = fd_analyzer.get_sets()
    fault_bin_sets = {k: v for k, v in fault_bin_sets.items() if len(v) > 0}

    if not fault_bin_sets:
        print("No fault diversity data available for plotting.")
        return

    print("\nStarting fault diversity plotting...")

    if from_contents is None or upset_plot is None:
        print("  [Skip] UpSetPlot is not installed; skipping fault diversity UpSet plot.")
    else:
        print(f"  Generating Fault Diversity UpSet Plot -> {OUT_FD_UPSET}")
        try:
            upset_data = from_contents(fault_bin_sets)
            fig = plt.figure(figsize=(11, 6))
            upset_plot(upset_data, subset_size='count', show_counts=True, sort_by='cardinality', fig=fig)
            plt.title(f"Intersection of Fault Diversity Bins (All Methods)", fontsize=16)
            plt.savefig(OUT_FD_UPSET, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"  [Error] Fault Diversity UpSet Plot: {e}")

    print(f"  Generating Fault Diversity Venn Diagram -> {OUT_FD_VENN}")
    if venn is None:
        print("  [Skip] venn is not installed; skipping fault diversity Venn plot.")
    else:
        try:
            plt.figure(figsize=(11, 11))
            venn(fault_bin_sets, cmap="plasma", alpha=0.3, legend_loc="upper right")
            plt.title(f"Venn Diagram of Fault Diversity Bins (All Methods)", fontsize=16)
            plt.savefig(OUT_FD_VENN, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"  [Error] Fault Diversity Venn Plot: {e}")

    print("\nDone. Please check the generated PDF files.")

if __name__ == "__main__":
    main()
