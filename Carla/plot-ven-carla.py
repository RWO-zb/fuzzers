import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from upsetplot import from_contents, plot as upset_plot
from venn import venn

# ================= 配置区域 =================
# 文件夹路径
DATA_DIR = "plot" 

# 文件名映射 (文件名 -> 图例标签)
# 已移除 "G-Model" 和 "Random"
FILE_MAP = {
    "curefuzz.csv": "CureFuzz",
    "mdpfuzz.csv":  "MDPFuzz",
    "qdfuzz.csv":   "QDFuzz",
    "seqfuzz.csv":  "SeqFuzz"
}

# 绘图参数
MAX_SEEDS = 1000       # 限制前 1000 个测试用例
TARGET_PHASE = "Phase2" # 仅统计 Phase2

# 输出文件名 (修改为 .pdf)
OUT_UPSET = "CARLA_Seed_Intersection_UpSet_NoRandG.pdf"
OUT_VENN = "CARLA_Seed_Intersection_Venn_NoRandG.pdf"

# 样式设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif', 
    'font.size': 12,
    'pdf.fonttype': 42, # 确保字体嵌入到 PDF 中，避免后续编辑缺失字体
    'ps.fonttype': 42
})

# ================= 数据处理类 =================

class CrashSeedAnalyzer:
    def __init__(self):
        self.data_store = {} # 存储处理后的 DataFrame

    def load_data(self):
        """加载数据并进行预处理（筛选 Phase, 截断 MAX_SEEDS）"""
        print(f"Loading data from {DATA_DIR}...")
        
        for fname, label in FILE_MAP.items():
            path = os.path.join(DATA_DIR, fname)
            # 兼容当前目录或 plot 目录
            if not os.path.exists(path):
                if os.path.exists(fname):
                    path = fname
                else:
                    print(f"  [Warn] File not found: {path}")
                    continue
                
            try:
                df = pd.read_csv(path)
                
                # --- 1. 标准化列名 ---
                df.columns = [c.strip() for c in df.columns]
                
                # --- 2. 筛选逻辑 ---
                df_filtered = pd.DataFrame()

                # 检查是否存在 Phase 列并进行筛选
                if 'phase' in df.columns and TARGET_PHASE in df['phase'].values:
                    df_filtered = df[df['phase'] == TARGET_PHASE].copy()
                else:
                    print(f"  [Info] {label}: '{TARGET_PHASE}' not found, using all data.")
                    df_filtered = df.copy()
                
                # --- 3. 数量截断 ---
                if MAX_SEEDS is not None and len(df_filtered) > MAX_SEEDS:
                    df_filtered = df_filtered.iloc[:MAX_SEEDS]
                
                self.data_store[label] = df_filtered
                print(f"  {label}: Loaded {len(df_filtered)} seeds.")

            except Exception as e:
                print(f"  [Error] Loading {label}: {e}")

    def compute_unique_seeds(self):
        """
        计算每个方法的唯一失效种子集合 (Success == False)
        集合元素定义: (Weather, Start_ID, Target_ID)
        """
        seed_sets = {}
        
        for label, df in self.data_store.items():
            if df.empty:
                continue

            # 1. 筛选失效用例
            if 'success' in df.columns:
                crashes = df[df['success'] == False].copy()
            elif 'collision' in df.columns:
                if df['collision'].dtype == object:
                    crashes = df[df['collision'].astype(str).str.lower() == 'true'].copy()
                else:
                    crashes = df[df['collision'] == True].copy()
            else:
                print(f"  [Warn] {label}: No 'success' or 'collision' column found.")
                continue
            
            if crashes.empty:
                seed_sets[label] = set()
                continue

            # 2. 提取种子标识
            unique_seeds = set()
            
            w_col = None
            if 'weather_id' in crashes.columns:
                w_col = 'weather_id'
            elif 'weather' in crashes.columns:
                w_col = 'weather'
            
            if 'start_id' in crashes.columns and 'target_id' in crashes.columns:
                for _, row in crashes.iterrows():
                    s_id = row['start_id']
                    t_id = row['target_id']
                    w_id = row[w_col] if w_col else -1
                    
                    seed_signature = (w_id, s_id, t_id)
                    unique_seeds.add(seed_signature)
            else:
                print(f"  [Error] {label}: Missing start_id or target_id.")
            
            seed_sets[label] = unique_seeds
            
        return seed_sets

# ================= 主程序 =================

def main():
    analyzer = CrashSeedAnalyzer()
    analyzer.load_data()
    
    crash_sets = analyzer.compute_unique_seeds()

    if not crash_sets:
        print("No crash data available for plotting.")
        return

    print("\nUnique Crash Seeds (Success=False) Found:")
    for k, v in crash_sets.items():
        print(f"  {k}: {len(v)}")

    # 1. 绘制 UpSet Plot
    print("\nGenerating UpSet Plot...")
    try:
        upset_data = from_contents(crash_sets)
        fig = plt.figure(figsize=(10, 6))
        upset_plot(upset_data, subset_size='count', show_counts=True, sort_by='cardinality', fig=fig)
        plt.title(f"Intersection of Crash Seeds (Fuzzers Only)", fontsize=16)
        
        # 保存为 PDF
        plt.savefig(OUT_UPSET, dpi=300, bbox_inches='tight')
        print(f"Saved {OUT_UPSET}")
    except Exception as e:
        print(f"UpSet Plot Error: {e}")

    # 2. 绘制 Venn 图
    print("\nGenerating Venn Diagram...")
    try:
        plt.figure(figsize=(9, 9))
        venn(crash_sets, cmap="plasma", alpha=0.3, legend_loc="upper right")
        plt.title(f"Venn Diagram of Crash Seeds (Fuzzers Only)", fontsize=16)
        
        # 保存为 PDF
        plt.savefig(OUT_VENN, dpi=300, bbox_inches='tight')
        print(f"Saved {OUT_VENN}")
    except Exception as e:
        print(f"Venn Plot Error: {e}")

    plt.show()

if __name__ == "__main__":
    main()