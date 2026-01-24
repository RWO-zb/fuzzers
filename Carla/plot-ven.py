import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from upsetplot import from_contents, plot as upset_plot
from venn import venn

# ================= 配置区域 =================
# 文件夹路径
DATA_DIR = "plot" 

# 文件名映射 (文件名 -> 图例标签)
# 自动匹配 plot 文件夹下的 csv 文件
FILE_MAP = {
    "curefuzz.csv": "CureFuzz",
    "g-model.csv":  "G-Model",
    "mdpfuzz.csv":  "MDPFuzz",
    "qdfuzz.csv":   "QDFuzz",
    "random.csv":   "Random",
    "seqfuzz.csv":  "SeqFuzz"
}

# 绘图参数
MAX_SEEDS = 1000       # 限制前 1000 个测试用例 (设为 None 则不限制)
GRID_BINS = (100, 100)   # 将地图划分为 50x50 的网格
TARGET_PHASE = "Phase2" # 仅统计 Phase2 (对于非 G-Model/Random 方法)

# 样式设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

# ================= 数据处理类 =================

class CrashGridAnalyzer:
    def __init__(self):
        self.all_x = []
        self.all_y = []
        self.data_store = {}

    def load_data(self):
        """加载所有 CSV 并收集坐标用于计算地图范围"""
        print(f"Loading data from {DATA_DIR}...")
        
        for fname, label in FILE_MAP.items():
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                print(f"  [Warn] File not found: {path}")
                continue
                
            try:
                df = pd.read_csv(path)
                
                # --- 1. 预处理与筛选 ---
                # 标准化列名 (去除空格)
                df.columns = [c.strip() for c in df.columns]
                
                # 筛选逻辑 (参考 RQ2)
                if label in ["G-Model"]:
                    # G-Model: 从 generative 阶段开始
                    if 'method' in df.columns and 'generative+novelty' in df['method'].values:
                        start_idx = df[df['method'] == 'generative+novelty'].index[0]
                        df = df.loc[start_idx:].copy()
                elif label in ["Random"]:
                    pass # Random 取全部
                else:
                    # 其他方法: 筛选 Phase2 (如果存在)
                    if 'phase' in df.columns and TARGET_PHASE in df['phase'].values:
                        df = df[df['phase'] == TARGET_PHASE].copy()
                
                # --- 2. 数量截断 (模拟 Budget) ---
                if MAX_SEEDS is not None and len(df) > MAX_SEEDS:
                    df = df.iloc[:MAX_SEEDS]
                
                # --- 3. 提取 Crash ---
                # 兼容不同的 True/False 写法
                if df['collision'].dtype == object:
                    crashes = df[df['collision'].astype(str).str.lower() == 'true'].copy()
                else:
                    crashes = df[df['collision'] == True].copy()
                
                if crashes.empty:
                    print(f"  {label}: No crashes found.")
                    continue

                # 收集坐标用于后续归一化
                # 确保有坐标列
                if 'final_x' in crashes.columns and 'final_y' in crashes.columns:
                    self.all_x.extend(crashes['final_x'].dropna().tolist())
                    self.all_y.extend(crashes['final_y'].dropna().tolist())
                    self.data_store[label] = crashes[['final_x', 'final_y']].dropna()
                    print(f"  {label}: {len(crashes)} crashes loaded.")
                else:
                    print(f"  [Error] {label} missing final_x/final_y columns.")

            except Exception as e:
                print(f"  [Error] Loading {label}: {e}")

    def compute_grids(self):
        """计算每个方法的 Unique Grid 集合"""
        if not self.all_x:
            return {}

        # 1. 计算地图边界 (Dynamic Range)
        min_x, max_x = min(self.all_x), max(self.all_x)
        min_y, max_y = min(self.all_y), max(self.all_y)
        
        # 增加一点 buffer 防止边界溢出
        margin = 5.0
        min_x -= margin; max_x += margin
        min_y -= margin; max_y += margin
        
        print(f"\nMap Range: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
        
        grid_sets = {}
        
        for label, df in self.data_store.items():
            unique_grids = set()
            for _, row in df.iterrows():
                # 映射到 Grid ID
                gx = int((row['final_x'] - min_x) / (max_x - min_x) * GRID_BINS[0])
                gy = int((row['final_y'] - min_y) / (max_y - min_y) * GRID_BINS[1])
                
                # 限制在 [0, BINS-1] 范围内
                gx = max(0, min(gx, GRID_BINS[0] - 1))
                gy = max(0, min(gy, GRID_BINS[1] - 1))
                
                unique_grids.add((gx, gy))
            
            grid_sets[label] = unique_grids
            
        return grid_sets

# ================= 主程序 =================

def main():
    analyzer = CrashGridAnalyzer()
    analyzer.load_data()
    crash_sets = analyzer.compute_grids()

    if not crash_sets:
        print("No crash data available for plotting.")
        return

    # 打印统计
    print("\nUnique Crash Grids Found:")
    for k, v in crash_sets.items():
        print(f"  {k}: {len(v)}")

    # 1. 绘制 UpSet Plot
    print("\nGenerating UpSet Plot...")
    try:
        upset_data = from_contents(crash_sets)
        fig = plt.figure(figsize=(12, 7))
        upset_plot(upset_data, subset_size='count', show_counts=True, sort_by='cardinality', fig=fig)
        plt.title(f"Intersection of Crash Locations (Grid {GRID_BINS[0]}x{GRID_BINS[1]})", fontsize=16)
        plt.savefig("CARLA_Grid_Diversity_UpSet.png", dpi=300, bbox_inches='tight')
        print("Saved CARLA_Grid_Diversity_UpSet.png")
    except Exception as e:
        print(f"UpSet Plot Error: {e}")

    # 2. 绘制 Venn 图
    print("\nGenerating Venn Diagram...")
    try:
        plt.figure(figsize=(10, 10))
        venn(crash_sets, cmap="plasma", alpha=0.3, legend_loc="upper right")
        plt.title(f"Venn Diagram of Crash Locations", fontsize=16)
        plt.savefig("CARLA_Grid_Diversity_Venn.png", dpi=300, bbox_inches='tight')
        print("Saved CARLA_Grid_Diversity_Venn.png")
    except Exception as e:
        print(f"Venn Plot Error: {e}")

    plt.show()

if __name__ == "__main__":
    main()