import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from upsetplot import from_contents, plot as upset_plot
from venn import venn

# ================= 配置区域 =================
# 确保安装: pip install upsetplot venn matplotlib

# 文件路径配置 (已移除 Random 和 G-Model)
FILE_PATHS = {
    "CureFuzz": {
        "obs": "obs_sequences.pkl",
        "log": "selection_log.pkl"
    },
    "MDPFuzz": {
        "obs": "MC_DQN_NoCov_5_0.01_0.1_0_7000it_obs.txt"
    },
    "QDFuzz": {
        "obs": "mc_test_obs.txt"
    },
    "SeqFuzz": {
        "obs": "all_episodes_obs.txt",
        "log": "all_run_seeds_0.pkl"
    }
}

# MountainCar 物理参数范围
RANGES = {
    'bd_pos': (-1.2, 0.6),
    'bd_speed': (0.0, 0.05)
}
GRID_SIZE = (50, 50)  # 网格大小

# ================= 数据解析类 =================

class DataParser:
    @staticmethod
    def load_pickle(path):
        if not os.path.exists(path):
            print(f"[Warn] File not found: {path}")
            return []
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def parse_seqfuzz(obs_path, log_path):
        obs_seqs = []
        if os.path.exists(obs_path):
            current_seq = []
            with open(obs_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if '######' in line:
                        if current_seq:
                            obs_seqs.append(np.array(current_seq))
                            current_seq = []
                    else:
                        try:
                            parts = line.strip(',').split(',')
                            vals = [float(p) for p in parts if p.strip()]
                            if len(vals) >= 2: current_seq.append(vals[:2])
                        except: continue
            if current_seq: obs_seqs.append(np.array(current_seq))
        
        logs = DataParser.load_pickle(log_path)
        data = []
        min_len = min(len(obs_seqs), len(logs))
        for i in range(min_len):
            is_crash = logs[i].get('crashed', False)
            data.append((obs_seqs[i], is_crash))
        return data

    @staticmethod
    def parse_curefuzz(obs_path, log_path):
        obs_seqs = DataParser.load_pickle(obs_path)
        logs = DataParser.load_pickle(log_path)
        data = []
        min_len = min(len(obs_seqs), len(logs))
        for i in range(min_len):
            is_crash = logs[i].get('did_crash', False)
            data.append((obs_seqs[i], is_crash))
        return data

    @staticmethod
    def parse_mdpfuzz_style(obs_path, skip_gen0=True):
        """
        解析 MDPFuzz 风格数据。
        skip_gen0=True 表示跳过 Generation 0 (初始种子)，与 RQ2 逻辑一致。
        """
        if not os.path.exists(obs_path):
            print(f"[Warn] File not found: {obs_path}")
            return []
        data = []
        current_info = None
        current_data = []
        with open(obs_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith("--- Test Case Info:"):
                    # 处理上一条记录
                    if current_info is not None:
                        gen = current_info.get('Generation', 0)
                        # 核心逻辑：如果 skip_gen0 为 True 且是第0代，则不添加
                        if not (skip_gen0 and gen == 0):
                            is_crash = current_info.get('Oracle', False)
                            data.append((np.array(current_data), is_crash))
                    # 开始新记录
                    try:
                        json_part = line.split("--- Test Case Info:")[1].split("---")[0].strip()
                        current_info = json.loads(json_part)
                        current_data = []
                    except: current_info = None
                else:
                    if current_info is not None:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2: current_data.append([float(parts[0]), float(parts[1])])
                        except: continue
            # 处理最后一条
            if current_info is not None:
                gen = current_info.get('Generation', 0)
                if not (skip_gen0 and gen == 0):
                    is_crash = current_info.get('Oracle', False)
                    data.append((np.array(current_data), is_crash))
        return data

    @staticmethod
    def parse_qdfuzz(obs_path):
        # QDFuzz 同样跳过 Gen 0
        return DataParser.parse_mdpfuzz_style(obs_path, skip_gen0=True)

# ================= 集合提取类 =================

class DiversitySetExtractor:
    def __init__(self):
        self.behavior_grid = GRID_SIZE

    def _get_grid_index(self, values, ranges, grid_size):
        indices = []
        for val, (min_val, max_val), bins in zip(values, ranges, grid_size):
            norm = (val - min_val) / (max_val - min_val) if max_val != min_val else 0
            idx = int(norm * bins)
            idx = np.clip(idx, 0, bins - 1)
            indices.append(idx)
        return tuple(indices)

    def _calc_bd(self, sequence):
        seq_arr = np.array(sequence)
        if len(seq_arr) == 0: return -1.2, 0.0
        if seq_arr.ndim == 1: seq_arr = seq_arr.reshape(-1, 2)
        positions = seq_arr[:, 0]
        velocities = seq_arr[:, 1]
        return np.max(positions), np.mean(np.abs(velocities))

    def get_crash_grid_set(self, data):
        """
        输入 data: List of (seq, is_crash)
        输出: Set of Grid IDs where crash occurred
        """
        crash_grids = set()
        
        for seq, is_crash in data:
            if is_crash:
                # 计算 Behavior Descriptor (BD)
                bd = self._calc_bd(seq)
                # 映射到 Grid ID
                bd_idx = self._get_grid_index(
                    bd,
                    (RANGES['bd_pos'], RANGES['bd_speed']),
                    self.behavior_grid
                )
                crash_grids.add(bd_idx)
                
        return crash_grids

# ================= 主程序 =================

def main():
    extractor = DiversitySetExtractor()
    crash_sets = {}
    
    print("=== Extracting Crash Diversity Sets (4 Methods) ===")

    # 1. CureFuzz
    print("Processing CureFuzz...")
    d = DataParser.parse_curefuzz(FILE_PATHS['CureFuzz']['obs'], FILE_PATHS['CureFuzz']['log'])
    crash_sets['CureFuzz'] = extractor.get_crash_grid_set(d)

    # 2. MDPFuzz (Skip Gen 0 = True)
    print("Processing MDPFuzz (skip_gen0=True)...")
    d = DataParser.parse_mdpfuzz_style(FILE_PATHS['MDPFuzz']['obs'], skip_gen0=True)
    crash_sets['MDPFuzz'] = extractor.get_crash_grid_set(d)

    # 3. QDFuzz (Skip Gen 0 = True)
    print("Processing QDFuzz (skip_gen0=True)...")
    d = DataParser.parse_qdfuzz(FILE_PATHS['QDFuzz']['obs'])
    crash_sets['QDFuzz'] = extractor.get_crash_grid_set(d)

    # 4. SeqFuzz
    print("Processing SeqFuzz...")
    d = DataParser.parse_seqfuzz(FILE_PATHS['SeqFuzz']['obs'], FILE_PATHS['SeqFuzz']['log'])
    crash_sets['SeqFuzz'] = extractor.get_crash_grid_set(d)

    # 打印统计信息
    print("\nUnique Crash Grids Found (Cardinality):")
    sorted_keys = sorted(crash_sets.keys())
    for k in sorted_keys:
        print(f"  {k}: {len(crash_sets[k])}")

    # ================= 绘图逻辑 =================

    # 1. UpSet Plot
    print("\nGenerating UpSet Plot...")
    try:
        # 检查是否所有集合都为空
        if all(len(s) == 0 for s in crash_sets.values()):
            print("[Error] No crashes found in any method. Check data paths.")
        else:
            upset_data = from_contents(crash_sets)
            fig = plt.figure(figsize=(10, 6))
            upset_plot(upset_data, subset_size='count', show_counts=True, sort_by='cardinality', fig=fig)
            plt.title("Intersection of Crash Faults (Excluding Init Seeds)")
            plt.savefig("MC_Diversity_UpSet_4Methods.png", dpi=300, bbox_inches='tight')
            print("Saved MC_Diversity_UpSet_4Methods.png")
    except Exception as e:
        print(f"UpSet Plot Error: {e}")

    # 2. Venn Diagram
    print("\nGenerating Venn Diagram...")
    try:
        if all(len(s) == 0 for s in crash_sets.values()):
            pass
        else:
            plt.figure(figsize=(8, 8))
            # 绘制韦恩图 (4个集合会生成经典的椭圆形状)
            venn(crash_sets, cmap="plasma", alpha=0.3, legend_loc="upper right")
            plt.title("Venn Diagram of Crash Faults")
            plt.savefig("MC_Diversity_Venn_4Methods.png", dpi=300, bbox_inches='tight')
            print("Saved MC_Diversity_Venn_4Methods.png")
    except Exception as e:
        print(f"Venn Plot Error: {e}")

    plt.show()

if __name__ == "__main__":
    main()