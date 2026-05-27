import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from venn import venn
except ModuleNotFoundError:
    venn = None

try:
    from upsetplot import from_contents, plot as upset_plot
except ModuleNotFoundError:
    from_contents = None
    upset_plot = None


BASE_DIR = "RQ2"
OUT_UPSET = os.path.join(BASE_DIR, "MC_Crash_Intersection_UpSet_AllMethods.pdf")
OUT_VENN = os.path.join(BASE_DIR, "MC_Crash_Intersection_Venn_AllMethods.pdf")
OUT_FD_UPSET = os.path.join(BASE_DIR, "MC_Fault_Diversity_UpSet_AllMethods.pdf")
OUT_FD_VENN = os.path.join(BASE_DIR, "MC_Fault_Diversity_Venn_AllMethods.pdf")

FILES_CONFIG = {
    "CureFuzz": {
        "obs": os.path.join(BASE_DIR, "obs_sequences.pkl"),
        "log": os.path.join(BASE_DIR, "selection_log.pkl"),
        "type": "pickle_curefuzz",
    },
    "G-Model": {
        "obs": os.path.join(BASE_DIR, "all_trajectories.pkl"),
        "log": os.path.join(BASE_DIR, "all_test_cases_log.pkl"),
        "type": "pickle_gmodel",
    },
    "MDPFuzz": {
        "obs": os.path.join(BASE_DIR, "MC_DQN_NoCov_5_0.01_0.1_0_7000it_obs.txt"),
        "type": "text_mdpfuzz",
    },
    "QDFuzz": {
        "obs": os.path.join(BASE_DIR, "mc_test_obs.txt"),
        "type": "text_mdpfuzz",
    },
    "Random": {
        "obs": os.path.join(BASE_DIR, "MC_DQN_RT_0_5000it_obs.txt"),
        "type": "text_random",
    },
    "SeqFuzz": {
        "obs": os.path.join(BASE_DIR, "all_episodes_obs.txt"),
        "log": os.path.join(BASE_DIR, "all_run_seeds_0.pkl"),
        "type": "text_seqfuzz",
    },
}

RANGES = {
    "bd_pos": (-1.2, 0.6),
    "bd_speed": (0.0, 0.05),
}
GRID_SIZE = (50, 50)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_pickle(path):
    if not hasattr(np, "_core"):
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
        sys.modules.setdefault("numpy._core.numeric", numpy_numeric)

    with open(path, "rb") as f:
        return pickle.load(f)


def parse_bool(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def normalize_key(key_raw):
    if isinstance(key_raw, bytes):
        return str(key_raw)
    if isinstance(key_raw, (list, tuple, np.ndarray)):
        return str(tuple(np.asarray(key_raw).tolist()))
    return str(key_raw)


def calc_bd(sequence):
    seq_arr = np.array(sequence)
    if len(seq_arr) == 0:
        return -1.2, 0.0
    if seq_arr.ndim == 1:
        seq_arr = seq_arr.reshape(-1, 2)
    positions = seq_arr[:, 0]
    velocities = seq_arr[:, 1]
    return float(np.max(positions)), float(np.mean(np.abs(velocities)))


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
        return RANGES["bd_pos"], RANGES["bd_speed"]

    min_b0, max_b0 = min(all_b0), max(all_b0) + 1e-5
    min_b1, max_b1 = min(all_b1), max(all_b1) + 1e-5
    print(f"\n[Global FD Range] Position: [{min_b0:.2f}, {max_b0:.2f}], Speed: [{min_b1:.4f}, {max_b1:.4f}]")
    return (min_b0, max_b0), (min_b1, max_b1)


class DataParser:
    @staticmethod
    def parse_delimited_obs(obs_path, skip_gen0=True):
        if not os.path.exists(obs_path):
            print(f"  [Warn] File not found: {obs_path}")
            return []

        data = []
        current_info = None
        current_data = []

        with open(obs_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("--- Test Case Info:"):
                    if current_info is not None:
                        DataParser._append_delimited_case(data, current_info, current_data, skip_gen0)

                    try:
                        json_part = line.split("--- Test Case Info:")[1].split("---")[0].strip()
                        current_info = json.loads(json_part)
                        current_data = []
                    except Exception:
                        current_info = None
                        current_data = []
                elif current_info is not None:
                    try:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            current_data.append([float(parts[0]), float(parts[1])])
                    except Exception:
                        continue

            if current_info is not None:
                DataParser._append_delimited_case(data, current_info, current_data, skip_gen0)

        return data

    @staticmethod
    def _append_delimited_case(data, info, sequence, skip_gen0):
        gen = info.get("Generation", 0)
        if skip_gen0 and gen == 0:
            return

        data.append({
            "input": info.get("Input"),
            "sequence": np.array(sequence),
            "is_crash": parse_bool(info.get("Oracle", False)),
        })

    @staticmethod
    def parse_seqfuzz(obs_path, log_path):
        if not os.path.exists(obs_path):
            print(f"  [Warn] File not found: {obs_path}")
            return []
        if not os.path.exists(log_path):
            print(f"  [Warn] File not found: {log_path}")
            return []

        obs_seqs = []
        current_seq = []
        with open(obs_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "######" in line:
                    if current_seq:
                        obs_seqs.append(np.array(current_seq))
                        current_seq = []
                else:
                    try:
                        parts = line.strip(",").split(",")
                        vals = [float(p) for p in parts if p.strip()]
                        if len(vals) >= 2:
                            current_seq.append(vals[:2])
                    except Exception:
                        continue

        if current_seq:
            obs_seqs.append(np.array(current_seq))

        logs = load_pickle(log_path)
        data = []
        for seq, log in zip(obs_seqs, logs):
            data.append({
                "input": log.get("state"),
                "sequence": seq,
                "is_crash": parse_bool(log.get("crashed", False)),
            })
        return data

    @staticmethod
    def parse_pickle_with_logs(obs_path, log_path, input_key, crash_key):
        if not os.path.exists(obs_path):
            print(f"  [Warn] File not found: {obs_path}")
            return []
        if not os.path.exists(log_path):
            print(f"  [Warn] File not found: {log_path}")
            return []

        obs_seqs = load_pickle(obs_path)
        logs = load_pickle(log_path)
        data = []
        for seq, log in zip(obs_seqs, logs):
            data.append({
                "input": log.get(input_key),
                "sequence": seq,
                "is_crash": parse_bool(log.get(crash_key, False)),
            })
        return data

    @staticmethod
    def load_method(config):
        file_type = config["type"]
        if file_type == "pickle_curefuzz":
            return DataParser.parse_pickle_with_logs(config["obs"], config["log"], "mutate_state", "did_crash")
        if file_type == "pickle_gmodel":
            return DataParser.parse_pickle_with_logs(config["obs"], config["log"], "input", "is_crash")
        if file_type == "text_mdpfuzz":
            return DataParser.parse_delimited_obs(config["obs"], skip_gen0=True)
        if file_type == "text_random":
            return DataParser.parse_delimited_obs(config["obs"], skip_gen0=False)
        if file_type == "text_seqfuzz":
            return DataParser.parse_seqfuzz(config["obs"], config["log"])
        return []


class CrashInputAnalyzer:
    def __init__(self):
        self.data_store = {}
        self.crash_sets = {}

    def load_data(self):
        print(f"Loading crash data from {BASE_DIR}...")
        for name, config in FILES_CONFIG.items():
            try:
                records = DataParser.load_method(config)
                self.data_store[name] = records
                print(f"  {name}: Loaded {len(records)} records.")
            except Exception as e:
                print(f"  [Error] {name}: {e}")

    def compute_unique_crashes(self):
        for name, records in self.data_store.items():
            unique_crashes = set()
            for record in records:
                if not record["is_crash"]:
                    continue
                inp = record.get("input")
                if inp is None:
                    inp = calc_bd(record["sequence"])
                unique_crashes.add(normalize_key(inp))

            self.crash_sets[name] = unique_crashes
            print(f"  {name}: Found {len(unique_crashes)} unique crash inputs.")

    def get_sets(self):
        return self.crash_sets


class FaultDiversityAnalyzer:
    def __init__(self):
        self.records = {}
        self.fault_bin_sets = {}

    def load_data(self):
        print("\nLoading behavior data for fault diversity...")
        for name, config in FILES_CONFIG.items():
            try:
                parsed = DataParser.load_method(config)
                records = []
                for record in parsed:
                    bd_pos, bd_speed = calc_bd(record["sequence"])
                    records.append({
                        "bd_0": bd_pos,
                        "bd_1": bd_speed,
                        "is_crash": record["is_crash"],
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
    analyzer = CrashInputAnalyzer()
    analyzer.load_data()
    analyzer.compute_unique_crashes()
    crash_sets = analyzer.get_sets()
    crash_sets = {k: v for k, v in crash_sets.items() if len(v) > 0}

    if not crash_sets:
        print("No crash data available for plotting.")
    else:
        print("\nStarting crash input plotting...")
        plot_upset(crash_sets, OUT_UPSET, "Intersection of Crash Inputs (All Methods)")
        plot_venn(crash_sets, OUT_VENN, "Venn Diagram of Crash Inputs (All Methods)")

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
