import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from upsetplot import from_contents, plot as upset_plot
from venn import venn

# ==========================================
# 1. 复用 plot-RQ2.py 的配置与加载逻辑
# ==========================================
FILES_CONFIG = {
    "CureFuzz": {"path": "selection_log.pkl", "type": "pickle_curefuzz"},
    "G-Model":  {"path": "all_test_cases_log.pkl", "type": "pickle_gmodel"},
    "MDPFuzz":  {"path": "fuzzer_10_0.01_0.01_0_logs.txt", "type": "csv_mdpfuzz"},
    "QDFuzz":   {"path": "1768120702.3916006_data.csv", "type": "csv_qdfuzz"}, 
    "Random":   {"path": "rt_10_0.01_0.01_1_logs.txt", "type": "csv_mdpfuzz"},
    "SeqFuzz":  {"path": "all_run_seeds_0.pkl", "type": "pickle_seqfuzz"}
}

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_crash_states(method_name, config):
    """
    仅提取发生 Crash 的唯一状态 (State Key)
    """
    path = config['path']
    file_type = config['type']
    crash_set = set()

    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return crash_set

    print(f"Loading {method_name} crashes from {path}...")
    
    try:
        # --- Copy logic from plot-RQ2.py but simplified for sets ---
        if file_type == 'pickle_curefuzz':
            data = load_pickle(path)
            for entry in data:
                if entry.get('did_crash', False):
                    state = entry.get('mutate_state')
                    key = state.tobytes() if hasattr(state, 'tobytes') else str(state)
                    crash_set.add(key)

        elif file_type == 'pickle_gmodel':
            data = load_pickle(path)
            for entry in data:
                if entry.get('is_crash', False):
                    inp = entry.get('input')
                    key = tuple(inp) if isinstance(inp, list) else (inp.tobytes() if hasattr(inp, 'tobytes') else str(inp))
                    crash_set.add(key)

        elif file_type == 'pickle_seqfuzz':
            data = load_pickle(path)
            for entry in data:
                if entry.get('crashed', False):
                    state = entry.get('state')
                    key = state.tobytes() if hasattr(state, 'tobytes') else str(state)
                    crash_set.add(key)

        elif file_type == 'csv_mdpfuzz': # Handles Random too
            df = pd.read_csv(path, delimiter=';', engine='python', on_bad_lines='skip', skipinitialspace=True)
            if 'Oracle' in df.columns:
                df['Oracle'] = df['Oracle'].astype(str).map({'True': True, 'False': False, 'None': False}).fillna(False)
            
            # Filter standard
            df = df[df['Oracle'] == True]
            for row in df.itertuples(index=False):
                inp = getattr(row, 'Input', None)
                if inp:
                    crash_set.add(str(inp))
        
        elif file_type == 'csv_qdfuzz':
            df = pd.read_csv(path)
            if 'is_faulty' in df.columns:
                df = df[df['is_faulty'] == True]
            for row in df.itertuples(index=False):
                inp = getattr(row, 'input', None)
                if inp:
                    crash_set.add(str(inp))
                    
    except Exception as e:
        print(f"Error loading {method_name}: {e}")
        
    print(f"  -> Found {len(crash_set)} unique crashes.")
    return crash_set

# ==========================================
# 2. 准备数据
# ==========================================

# 提取每个方法的 Crash 集合
crash_sets_dict = {}
for name, config in FILES_CONFIG.items():
    crash_sets_dict[name] = load_crash_states(name, config)

# ==========================================
# 3. 绘制 UpSet Plot (推荐)
# ==========================================
print("\nGenerating UpSet Plot...")
plt.figure(figsize=(12, 8))

# 转换数据格式
upset_data = from_contents(crash_sets_dict)

# 绘图
# subset_size='count' 显示交集数量
# show_counts=True 显示具体数字
upset_plot(upset_data, subset_size='count', show_counts=True, sort_by='cardinality')

plt.title("Intersection of Unique Crashes (Fault Diversity)")
plt.savefig("Diversity_UpSet.png", dpi=300)
print("Saved to Diversity_UpSet.png")
plt.close()

# ==========================================
# 4. 绘制 6集合 韦恩图 (形状图)
# ==========================================
print("\nGenerating Venn Diagram...")
plt.figure(figsize=(10, 10))

# 使用 venn 库绘制
# legend_loc="upper right" 调整图例位置
venn(crash_sets_dict, cmap="plasma", alpha=0.3, legend_loc="upper right")

plt.title("Venn Diagram of Fault Diversity")
plt.savefig("Diversity_Venn.png", dpi=300)
print("Saved to Diversity_Venn.png")
plt.close()