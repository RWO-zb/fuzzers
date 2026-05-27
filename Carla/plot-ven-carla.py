import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
TARGET_PHASE = "Phase2"
MAX_SEEDS = 1000
GRID_SIZE = (10, 10)

OUT_UPSET = os.path.join(BASE_DIR, "CARLA_Crash_Intersection_UpSet_Seed0_AllMethods.pdf")
OUT_VENN = os.path.join(BASE_DIR, "CARLA_Crash_Intersection_Venn_Seed0_AllMethods.pdf")
OUT_FD_UPSET = os.path.join(BASE_DIR, "CARLA_Fault_Diversity_UpSet_Seed0_AllMethods.pdf")
OUT_FD_VENN = os.path.join(BASE_DIR, "CARLA_Fault_Diversity_Venn_Seed0_AllMethods.pdf")

FILES_CONFIG = {
    "CureFuzz": {
        "path": os.path.join(BASE_DIR, "curefuzz.csv"),
        "label": "CureFuzz",
    },
    "G-Model": {
        "path": os.path.join(BASE_DIR, "g-model.csv"),
        "label": "G-Model",
    },
    "MDPFuzz": {
        "path": os.path.join(BASE_DIR, "mdpfuzz.csv"),
        "label": "MDPFuzz",
    },
    "QDFuzz": {
        "path": os.path.join(BASE_DIR, "qdfuzz.csv"),
        "label": "QDFuzz",
    },
    "Random": {
        "path": os.path.join(BASE_DIR, "random.csv"),
        "label": "Random",
    },
    "SeqFuzz": {
        "path": os.path.join(BASE_DIR, "seqfuzz.csv"),
        "label": "SeqFuzz",
    },
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def parse_bool(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def normalize_key(key_raw):
    if isinstance(key_raw, bytes):
        return str(key_raw)
    if isinstance(key_raw, (list, tuple, np.ndarray)):
        return str(tuple(key_raw))
    return str(key_raw)


def get_bin_index(value, min_val, max_val, grid_dim):
    if max_val <= min_val:
        return 0
    idx = int((value - min_val) / (max_val - min_val) * grid_dim)
    return min(max(idx, 0), grid_dim - 1)


def load_method_frame(path, label):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "phase" in df.columns and TARGET_PHASE in df["phase"].values:
        df = df[df["phase"] == TARGET_PHASE].copy()
    elif "phase" in df.columns:
        print(f"  [Info] {label}: '{TARGET_PHASE}' not found, using all data.")
        df = df.copy()
    else:
        df = df.copy()

    if MAX_SEEDS is not None and len(df) > MAX_SEEDS:
        df = df.iloc[:MAX_SEEDS].copy()

    return df


def get_crash_mask(df):
    if "success" in df.columns:
        return df["success"].map(parse_bool) == False
    if "collision" in df.columns:
        return df["collision"].map(parse_bool) == True
    return None


def get_weather_column(df):
    if "weather_id" in df.columns:
        return "weather_id"
    if "weather" in df.columns:
        return "weather"
    return None


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
    print(f"\n[Global FD Range] Avg Speed: [{min_b0:.2f}, {max_b0:.2f}], Steer Std: [{min_b1:.2f}, {max_b1:.2f}]")
    return (min_b0, max_b0), (min_b1, max_b1)


class CrashSeedAnalyzer:
    def __init__(self):
        self.data_store = {}
        self.crash_sets = {}

    def load_data(self):
        print(f"Loading crash data from {BASE_DIR}...")

        for name, config in FILES_CONFIG.items():
            path = config["path"]

            if not os.path.exists(path):
                print(f"  [Warn] {name}: File not found ({path})")
                continue

            try:
                df = load_method_frame(path, config["label"])
                self.data_store[name] = df
                print(f"  {name}: Loaded {len(df)} seeds.")
            except Exception as e:
                print(f"  [Error] Loading {name}: {e}")

    def compute_unique_seeds(self):
        for label, df in self.data_store.items():
            if df.empty:
                continue

            crash_mask = get_crash_mask(df)
            if crash_mask is None:
                print(f"  [Warn] {label}: No 'success' or 'collision' column found.")
                continue

            crashes = df[crash_mask].copy()
            unique_seeds = set()

            if crashes.empty:
                self.crash_sets[label] = unique_seeds
                continue

            w_col = get_weather_column(crashes)
            if "start_id" not in crashes.columns or "target_id" not in crashes.columns:
                print(f"  [Error] {label}: Missing start_id or target_id.")
                continue

            for row in crashes.itertuples(index=False):
                w_id = getattr(row, w_col) if w_col else -1
                seed_signature = (w_id, getattr(row, "start_id"), getattr(row, "target_id"))
                unique_seeds.add(normalize_key(seed_signature))

            self.crash_sets[label] = unique_seeds
            print(f"  {label}: Found {len(unique_seeds)} unique crash seeds.")

    def get_sets(self):
        return self.crash_sets


class FaultDiversityAnalyzer:
    def __init__(self):
        self.records = {}
        self.fault_bin_sets = {}

    def load_data(self):
        print("\nLoading behavior data for fault diversity...")

        for name, config in FILES_CONFIG.items():
            path = config["path"]

            if not os.path.exists(path):
                print(f"  [Warn] {name}: File not found ({path})")
                continue

            records = []

            try:
                df = load_method_frame(path, config["label"])
                crash_mask = get_crash_mask(df)
                if crash_mask is None:
                    print(f"  [Warn] {name}: No 'success' or 'collision' column found.")
                    continue

                for col in ["avg_speed", "steer_std"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                if "avg_speed" not in df.columns or "steer_std" not in df.columns:
                    print(f"  [Warn] {name}: Missing avg_speed or steer_std.")
                    continue

                df = df.dropna(subset=["avg_speed", "steer_std"])
                crash_mask = crash_mask.reindex(df.index)

                for row_idx, row in df.iterrows():
                    records.append({
                        "bd_0": float(row["avg_speed"]),
                        "bd_1": float(row["steer_std"]),
                        "is_crash": parse_bool(crash_mask.loc[row_idx]),
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


def plot_upset(sets, out_path, title):
    if from_contents is None or upset_plot is None:
        print(f"  [Skip] UpSetPlot is not installed; skipping {title}.")
        return

    print(f"  Generating UpSet Plot -> {out_path}")
    try:
        upset_data = from_contents(sets)
        fig = plt.figure(figsize=(11, 6))
        upset_plot(upset_data, subset_size="count", show_counts=True, sort_by="cardinality", fig=fig)
        plt.title(title, fontsize=16)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [Error] UpSet Plot: {e}")


def plot_venn(sets, out_path, title):
    print(f"  Generating Venn Diagram -> {out_path}")
    if venn is None:
        print(f"  [Skip] venn is not installed; skipping {title}.")
        return

    try:
        fig = plt.figure(figsize=(11, 11))
        venn(sets, cmap="plasma", alpha=0.3, legend_loc="upper right")
        plt.title(title, fontsize=16)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [Error] Venn Plot: {e}")


def main():
    analyzer = CrashSeedAnalyzer()
    analyzer.load_data()
    analyzer.compute_unique_seeds()
    crash_sets = analyzer.get_sets()
    crash_sets = {k: v for k, v in crash_sets.items() if len(v) > 0}

    if not crash_sets:
        print("No crash data available for plotting.")
    else:
        print("\nStarting crash seed plotting...")
        plot_upset(crash_sets, OUT_UPSET, "Intersection of Crash Seeds (All Methods)")
        plot_venn(crash_sets, OUT_VENN, "Venn Diagram of Crash Seeds (All Methods)")

    fd_analyzer = FaultDiversityAnalyzer()
    fd_analyzer.load_data()
    fd_analyzer.compute_fault_bins()
    fault_bin_sets = fd_analyzer.get_sets()
    fault_bin_sets = {k: v for k, v in fault_bin_sets.items() if len(v) > 0}

    if not fault_bin_sets:
        print("No fault diversity data available for plotting.")
        return

    print("\nStarting fault diversity plotting...")
    plot_upset(fault_bin_sets, OUT_FD_UPSET, "Intersection of Fault Diversity Bins (All Methods)")
    plot_venn(fault_bin_sets, OUT_FD_VENN, "Venn Diagram of Fault Diversity Bins (All Methods)")

    print("\nDone. Please check the generated PDF files.")


if __name__ == "__main__":
    main()
