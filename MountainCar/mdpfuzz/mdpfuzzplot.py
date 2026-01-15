import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 (请在此处修改文件名) =================
# 1. MDPFuzz 的日志文件路径 (请修改为您实际的文件路径)
MDPFUZZ_FILE = 'logs/MC_DQN_NoCov_5_0.01_0.1_0_7000it_obs.txt' 

# 2. Random Testing 的日志文件路径 (请修改为您实际的文件路径)
RANDOM_FILE  = 'logs/MC_DQN_RT_0_5000it_obs.txt'

# 3. 输出图片文件名
PLOT_FILENAME = 'diversity_comparison.png'
# ===============================================================

# --- 全局绘图风格 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5
})

class MdpFuzzLogParser:
    """解析 *_obs.txt 文件"""
    def __init__(self, filepath):
        self.filepath = filepath

    def parse(self):
        """生成器：逐个返回 (info_dict, obs_seq_array)"""
        if not os.path.exists(self.filepath):
            print(f"[Error] File not found: {self.filepath}")
            return

        print(f"Parsing: {self.filepath} ...")
        count = 0
        with open(self.filepath, 'r') as f:
            current_info = None
            current_data = []
            
            for line in f:
                line = line.strip()
                if not line: continue
                
                if line.startswith("--- Test Case Info:"):
                    # 如果之前有正在读取的 Case，先 yield 出来
                    if current_info is not None:
                        yield current_info, np.array(current_data)
                        count += 1
                        if count % 2000 == 0:
                            print(f"  Processed {count} episodes...", end='\r')
                    
                    # 解析新的 Header
                    try:
                        json_part = line[len("--- Test Case Info: "):-len(" ---")]
                        current_info = json.loads(json_part)
                        current_data = []
                    except Exception:
                        current_info = None
                else:
                    # 解析数据行 (Pos, Vel)
                    if current_info is not None:
                        try:
                            vals = [float(x) for x in line.split(',')]
                            current_data.append(vals)
                        except ValueError:
                            pass
            
            # Yield 最后一个 Case
            if current_info is not None:
                yield current_info, np.array(current_data)
        print(f"\nFinished. Total episodes: {count + 1}")

class DiversityAnalyzer:
    """计算指标: MaxPos × AvgSpeed"""
    def __init__(self):
        # 定义网格大小
        self.state_grid_size = (50, 50)
        self.behavior_grid_size = (50, 50)
        
        # 定义物理边界
        self.state_range = [(-1.2, 0.6), (-0.07, 0.07)] # Position, Velocity
        self.bd_range = [(-1.2, 0.6), (0.0, 0.05)]      # MaxPos, AvgSpeed

    def _get_grid_index(self, values, ranges, sizes):
        indices = []
        for val, (min_v, max_v), size in zip(values, ranges, sizes):
            norm = (val - min_v) / (max_v - min_v)
            idx = int(norm * size)
            idx = np.clip(idx, 0, size - 1)
            indices.append(idx)
        return tuple(indices)

    def _calc_bd(self, seq_arr):
        if len(seq_arr) == 0: return -1.2, 0.0
        max_pos = np.max(seq_arr[:, 0])
        avg_speed = np.mean(np.abs(seq_arr[:, 1]))
        return max_pos, avg_speed

    def analyze(self, parser):
        visited_state = set()
        visited_behavior = set()
        visited_fault = set()

        history = {
            'episodes': [],
            'state_coverage': [],
            'behavior_diversity': [],
            'fault_diversity': []
        }

        idx = 0
        for info, obs_seq in parser.parse():
            idx += 1
            is_crash = info.get('Oracle', False)

            # 1. State Coverage
            for state in obs_seq:
                if len(state) >= 2:
                    grid_idx = self._get_grid_index(
                        (state[0], state[1]), self.state_range, self.state_grid_size
                    )
                    visited_state.add(grid_idx)

            # 2. Behavior / Fault Diversity
            bd_val = self._calc_bd(obs_seq)
            bd_idx = self._get_grid_index(bd_val, self.bd_range, self.behavior_grid_size)
            
            visited_behavior.add(bd_idx)
            if is_crash:
                visited_fault.add(bd_idx)

            # 降采样记录 (每10个点记录一次，避免绘图过慢)
            if idx % 10 == 0:
                history['episodes'].append(idx)
                history['state_coverage'].append(len(visited_state))
                history['behavior_diversity'].append(len(visited_behavior))
                history['fault_diversity'].append(len(visited_fault))
        
        # 确保记录最后一点
        if not history['episodes'] or history['episodes'][-1] != idx:
            history['episodes'].append(idx)
            history['state_coverage'].append(len(visited_state))
            history['behavior_diversity'].append(len(visited_behavior))
            history['fault_diversity'].append(len(visited_fault))
            
        return history

def plot_all(results_dict, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 定义三个子图的配置
    configs = [
        ("state_coverage", "State Coverage", "Position × Velocity"),
        ("behavior_diversity", "Behavior Diversity", "MaxPos × AvgSpeed"),
        ("fault_diversity", "Fault Diversity", "Crash Scenarios")
    ]

    for i, (key, title, subtitle) in enumerate(configs):
        ax = axes[i]
        
        for label, data in results_dict.items():
            if data is None: continue
            
            # 设置颜色: MDPFuzz为红色, Random为蓝色
            color = "#d62728" if "MDPFuzz" in label else "#1f77b4"
            
            ax.plot(data['episodes'], data[key], label=label, color=color, alpha=0.9)
            # 添加浅色填充
            ax.fill_between(data['episodes'], data[key], color=color, alpha=0.1)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Unique Bins Count")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 说明文字
        ax.text(0.02, 0.95, subtitle, transform=ax.transAxes, 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        if i == 0:
            ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")
    plt.show()

def main():
    analyzer = DiversityAnalyzer()
    results = {}

    # 1. 尝试加载 MDPFuzz 数据
    if os.path.exists(MDPFUZZ_FILE):
        print(f"Loading MDPFuzz data from: {MDPFUZZ_FILE}")
        parser = MdpFuzzLogParser(MDPFUZZ_FILE)
        results["MDPFuzz"] = analyzer.analyze(parser)
    else:
        print(f"[Warning] MDPFuzz file not found (Skipping): {MDPFUZZ_FILE}")

    # 2. 尝试加载 Random Testing 数据
    if os.path.exists(RANDOM_FILE):
        print(f"Loading Random data from: {RANDOM_FILE}")
        parser = MdpFuzzLogParser(RANDOM_FILE)
        results["Random Testing"] = analyzer.analyze(parser)
    else:
        print(f"[Warning] Random file not found (Skipping): {RANDOM_FILE}")

    # 3. 绘图
    if results:
        plot_all(results, PLOT_FILENAME)
    else:
        print("No valid data files found. Please check paths in '配置区域'.")

if __name__ == "__main__":
    main()