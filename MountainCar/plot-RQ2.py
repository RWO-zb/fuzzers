import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 输出文件名
PLOT_FILE = 'Combined_RQ2_Diversity.png'

# 各方法的数据文件路径 (根据提供的各个脚本中的路径整理)
# 请确保这些文件存在于对应的目录中
FILE_PATHS = {
    "CureFuzz": {
        "obs": "obs_sequences.pkl",
        "log": "selection_log.pkl"
    },
    "G-Model": {
        "traj": "all_trajectories.pkl",
        "log": "all_test_cases_log.pkl"
    },
    "MDPFuzz": {
        "obs": "MC_DQN_NoCov_5_0.01_0.1_0_7000it_obs.txt"
    },
    "QDFuzz": {
        "obs": "mc_test_obs.txt"
    },
    "Random": {
        "obs": "MC_DQN_RT_0_5000it_obs.txt"
    },
    "SeqFuzz": {
        "obs": "all_episodes_obs.txt",
        "log": "all_run_seeds_0.pkl"
    }
}

# 绘图样式配置
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

# 颜色映射 (与 plot-RQ2.py 的顺序对应的默认颜色循环)
# Order: CureFuzz, G-Model, MDPFuzz, QDFuzz, Random, SeqFuzz
METHOD_COLORS = {
    "CureFuzz": "#1f77b4",  # Blue
    "G-Model":  "#ff7f0e",  # Orange
    "MDPFuzz":  "#2ca02c",  # Green
    "QDFuzz":   "#d62728",  # Red
    "Random":   "#9467bd",  # Purple
    "SeqFuzz":  "#8c564b"   # Brown
}

# MountainCar 物理参数范围
RANGES = {
    'state_pos': (-1.2, 0.6),
    'state_vel': (-0.07, 0.07),
    'bd_pos': (-1.2, 0.6),
    'bd_speed': (0.0, 0.05)
}

GRID_SIZE = (50, 50)  # 统一使用 50x50 网格

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
        # SeqFuzz 逻辑: TXT obs + PKL logs
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
    def parse_gmodel(traj_path, log_path):
        # G-Model 逻辑: PKL trajectories + PKL logs
        traj_seqs = DataParser.load_pickle(traj_path)
        logs = DataParser.load_pickle(log_path)
        
        data = []
        min_len = min(len(traj_seqs), len(logs))
        for i in range(min_len):
            is_crash = logs[i].get('is_crash', False)
            data.append((traj_seqs[i], is_crash))
        return data

    @staticmethod
    def parse_curefuzz(obs_path, log_path):
        # CureFuzz 逻辑: PKL obs + PKL logs
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
        # MDPFuzz & Random 逻辑: TXT with JSON info
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
                    # Process previous episode
                    if current_info is not None:
                        gen = current_info.get('Generation', 0)
                        if not (skip_gen0 and gen == 0):
                            is_crash = current_info.get('Oracle', False)
                            data.append((np.array(current_data), is_crash))
                    
                    # Start new episode
                    try:
                        json_part = line.split("--- Test Case Info:")[1].split("---")[0].strip()
                        current_info = json.loads(json_part)
                        current_data = []
                    except:
                        current_info = None
                else:
                    if current_info is not None:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                current_data.append([float(parts[0]), float(parts[1])])
                        except: continue
                        
            # Last episode
            if current_info is not None:
                gen = current_info.get('Generation', 0)
                if not (skip_gen0 and gen == 0):
                    is_crash = current_info.get('Oracle', False)
                    data.append((np.array(current_data), is_crash))
        return data

    @staticmethod
    def parse_qdfuzz(obs_path):
        # QDFuzz 逻辑: 类似于 MDPFuzz，跳过 Gen 0
        return DataParser.parse_mdpfuzz_style(obs_path, skip_gen0=True)

# ================= 核心分析类 =================

class DiversityAnalyzer:
    def __init__(self):
        self.state_grid = GRID_SIZE
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
        # 处理可能的形状问题
        if seq_arr.ndim == 1: seq_arr = seq_arr.reshape(-1, 2)
        
        positions = seq_arr[:, 0]
        velocities = seq_arr[:, 1]
        return np.max(positions), np.mean(np.abs(velocities))

    def calculate_metrics(self, data):
        """
        data: List of (sequence_array, is_crash_bool)
        """
        visited_state = set()
        visited_behavior = set()
        visited_fault = set()

        history = {
            'episodes': [],
            'state_coverage': [],
            'behavior_diversity': [],
            'fault_diversity': []
        }

        for i, (seq, is_crash) in enumerate(data):
            # State Coverage
            for state in seq:
                if len(state) >= 2:
                    idx = self._get_grid_index(
                        (state[0], state[1]),
                        (RANGES['state_pos'], RANGES['state_vel']),
                        self.state_grid
                    )
                    visited_state.add(idx)

            # Behavior Diversity
            bd = self._calc_bd(seq)
            bd_idx = self._get_grid_index(
                bd,
                (RANGES['bd_pos'], RANGES['bd_speed']),
                self.behavior_grid
            )
            visited_behavior.add(bd_idx)

            # Fault Diversity
            if is_crash:
                visited_fault.add(bd_idx)

            history['episodes'].append(i + 1)
            history['state_coverage'].append(len(visited_state))
            history['behavior_diversity'].append(len(visited_behavior))
            history['fault_diversity'].append(len(visited_fault))
            
        return history

# ================= [新增] 打印统计函数 =================
def print_final_stats(all_histories):
    """
    打印每种方法的最终多样性统计计数 (取 history 列表的最后一个值)
    """
    print("\n" + "="*65)
    print(f"{'Method':<15} | {'State Cov':<12} | {'Behav Div':<12} | {'Fault Div':<12}")
    print("-" * 65)
    
    # 对方法名进行排序，保证输出顺序稳定
    sorted_names = sorted(all_histories.keys())
    
    for name in sorted_names:
        history = all_histories[name]
        
        # 你的 analyzer 返回的字典键名是: 'state_coverage', 'behavior_diversity', 'fault_diversity'
        # 获取列表最后一个元素作为最终计数，如果列表为空则为0
        
        sc = history['state_coverage'][-1] if history['state_coverage'] else 0
        bd = history['behavior_diversity'][-1] if history['behavior_diversity'] else 0
        fd = history['fault_diversity'][-1] if history['fault_diversity'] else 0
        
        print(f"{name:<15} | {sc:<12} | {bd:<12} | {fd:<12}")
    
    print("="*65 + "\n")

# ================= 主程序 =================

def main():
    analyzer = DiversityAnalyzer()
    all_histories = {}

    print("=== Loading Data ===")
    
    # 1. CureFuzz
    print(f"Loading CureFuzz...")
    data_cure = DataParser.parse_curefuzz(FILE_PATHS['CureFuzz']['obs'], FILE_PATHS['CureFuzz']['log'])
    all_histories['CureFuzz'] = analyzer.calculate_metrics(data_cure)

    # 2. G-Model
    print(f"Loading G-Model...")
    data_g = DataParser.parse_gmodel(FILE_PATHS['G-Model']['traj'], FILE_PATHS['G-Model']['log'])
    all_histories['G-Model'] = analyzer.calculate_metrics(data_g)

    # 3. MDPFuzz
    print(f"Loading MDPFuzz...")
    data_mdp = DataParser.parse_mdpfuzz_style(FILE_PATHS['MDPFuzz']['obs'], skip_gen0=True)
    all_histories['MDPFuzz'] = analyzer.calculate_metrics(data_mdp)

    # 4. QDFuzz
    print(f"Loading QDFuzz...")
    data_qd = DataParser.parse_qdfuzz(FILE_PATHS['QDFuzz']['obs'])
    all_histories['QDFuzz'] = analyzer.calculate_metrics(data_qd)

    # 5. Random
    print(f"Loading Random...")
    data_rnd = DataParser.parse_mdpfuzz_style(FILE_PATHS['Random']['obs'], skip_gen0=False)
    all_histories['Random'] = analyzer.calculate_metrics(data_rnd)

    # 6. SeqFuzz
    print(f"Loading SeqFuzz...")
    data_seq = DataParser.parse_seqfuzz(FILE_PATHS['SeqFuzz']['obs'], FILE_PATHS['SeqFuzz']['log'])
    all_histories['SeqFuzz'] = analyzer.calculate_metrics(data_seq)

    # ================= [新增] 打印统计表格 =================
    print_final_stats(all_histories)
    # ======================================================

    # ================= 绘图 =================
    print("=== Plotting ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        {'key': 'state_coverage', 'title': 'State Coverage', 'ylabel': '# Unique State Bins'},
        {'key': 'behavior_diversity', 'title': 'Behavior Diversity', 'ylabel': '# Unique Behaviors'},
        {'key': 'fault_diversity', 'title': 'Fault Diversity', 'ylabel': '# Unique Faults'}
    ]

    # 按照字母顺序或 plot-RQ2 的配置顺序遍历方法
    method_order = ['CureFuzz', 'G-Model', 'MDPFuzz', 'QDFuzz', 'Random', 'SeqFuzz']

    for i, m_config in enumerate(metrics):
        ax = axes[i]
        key = m_config['key']
        
        for method_name in method_order:
            history = all_histories.get(method_name)
            if not history or not history['episodes']:
                continue
                
            ax.plot(history['episodes'], history[key], 
                    label=method_name, 
                    color=METHOD_COLORS[method_name],
                    alpha=0.9, linewidth=2)

        ax.set_title(m_config['title'], fontweight='bold', pad=10)
        ax.set_xlabel('Episodes')
        ax.set_ylabel(m_config['ylabel'])
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 只在第一张图显示图例
        if i == 0:
            ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    print(f"Plot saved to {PLOT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()