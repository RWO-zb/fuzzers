import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from venn import venn

# ================= 配置区域 =================

# 绘图样式设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

OUTPUT_FILE = "Combined_Venn_Diagram_Fixed.pdf"
CMAP_NAME = "plasma"  # 配色方案
ALPHA_VAL = 0.3       # 透明度

# ================= 1. MountainCar 数据处理模块 (mc/) =================

class MountainCarLoader:
    BASE_DIR = "mc"
    FILE_CONFIG = {
        "CureFuzz": {"obs": "obs_sequences.pkl", "log": "selection_log.pkl"},
        "MDPFuzz": {"obs": "MC_DQN_NoCov_5_0.01_0.1_0_7000it_obs.txt"},
        "QDFuzz": {"obs": "mc_test_obs.txt"},
        "SeqFuzz": {"obs": "all_episodes_obs.txt", "log": "all_run_seeds_0.pkl"}
    }
    RANGES = {'bd_pos': (-1.2, 0.6), 'bd_speed': (0.0, 0.05)}
    GRID_SIZE = (50, 50)

    @staticmethod
    def _get_path(filename):
        path = os.path.join(MountainCarLoader.BASE_DIR, filename)
        if not os.path.exists(path):
            print(f"  [Warn] MC File missing: {path}")
        return path

    @staticmethod
    def load_pickle(path):
        if not os.path.exists(path): return []
        with open(path, 'rb') as f: return pickle.load(f)

    @staticmethod
    def parse_seqfuzz(obs_file, log_file):
        obs_path = MountainCarLoader._get_path(obs_file)
        log_path = MountainCarLoader._get_path(log_file)
        
        obs_seqs = []
        if os.path.exists(obs_path):
            current_seq = []
            with open(obs_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if '######' in line:
                        if current_seq: obs_seqs.append(np.array(current_seq)); current_seq = []
                    else:
                        try:
                            parts = line.strip(',').split(',')
                            vals = [float(p) for p in parts if p.strip()]
                            if len(vals) >= 2: current_seq.append(vals[:2])
                        except: continue
            if current_seq: obs_seqs.append(np.array(current_seq))
        
        logs = MountainCarLoader.load_pickle(log_path)
        data = []
        for i in range(min(len(obs_seqs), len(logs))):
            data.append((obs_seqs[i], logs[i].get('crashed', False)))
        return data

    @staticmethod
    def parse_curefuzz(obs_file, log_file):
        obs_path = MountainCarLoader._get_path(obs_file)
        log_path = MountainCarLoader._get_path(log_file)
        obs_seqs = MountainCarLoader.load_pickle(obs_path)
        logs = MountainCarLoader.load_pickle(log_path)
        data = []
        for i in range(min(len(obs_seqs), len(logs))):
            data.append((obs_seqs[i], logs[i].get('did_crash', False)))
        return data

    @staticmethod
    def parse_mdpfuzz_style(obs_file, skip_gen0=True):
        obs_path = MountainCarLoader._get_path(obs_file)
        if not os.path.exists(obs_path): return []
        data = []
        current_info = None
        current_data = []
        with open(obs_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith("--- Test Case Info:"):
                    if current_info is not None:
                        gen = current_info.get('Generation', 0)
                        if not (skip_gen0 and gen == 0):
                            data.append((np.array(current_data), current_info.get('Oracle', False)))
                    try:
                        current_info = json.loads(line.split("--- Test Case Info:")[1].split("---")[0].strip())
                        current_data = []
                    except: current_info = None
                else:
                    if current_info is not None:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2: current_data.append([float(parts[0]), float(parts[1])])
                        except: continue
            if current_info is not None:
                gen = current_info.get('Generation', 0)
                if not (skip_gen0 and gen == 0):
                    data.append((np.array(current_data), current_info.get('Oracle', False)))
        return data

    @staticmethod
    def get_crash_grid_set(data):
        crash_grids = set()
        for seq, is_crash in data:
            if is_crash:
                # ================= 修复部分 =================
                # 强制转换为 numpy array，解决 'list' has no attribute 'ndim' 错误
                try:
                    seq = np.array(seq)
                except Exception:
                    continue # 如果数据无法转换，跳过
                
                if len(seq) == 0: continue
                
                # 确保维度正确
                if seq.ndim == 1: 
                    # 尝试 reshape，如果元素数量不是偶数可能会报错，加个保护
                    try:
                        seq = seq.reshape(-1, 2)
                    except:
                        continue 
                # ===========================================

                bd = (np.max(seq[:, 0]), np.mean(np.abs(seq[:, 1])))
                idx = []
                for val, (min_v, max_v), bins in zip(bd, [MountainCarLoader.RANGES['bd_pos'], MountainCarLoader.RANGES['bd_speed']], MountainCarLoader.GRID_SIZE):
                    norm = (val - min_v) / (max_v - min_v) if max_v != min_v else 0
                    idx.append(np.clip(int(norm * bins), 0, bins - 1))
                crash_grids.add(tuple(idx))
        return crash_grids

    @classmethod
    def get_data(cls):
        print("Processing MountainCar (mc/)...")
        sets = {}
        try:
            sets['CureFuzz'] = cls.get_crash_grid_set(cls.parse_curefuzz(cls.FILE_CONFIG['CureFuzz']['obs'], cls.FILE_CONFIG['CureFuzz']['log']))
            sets['MDPFuzz'] = cls.get_crash_grid_set(cls.parse_mdpfuzz_style(cls.FILE_CONFIG['MDPFuzz']['obs'], skip_gen0=True))
            sets['QDFuzz'] = cls.get_crash_grid_set(cls.parse_mdpfuzz_style(cls.FILE_CONFIG['QDFuzz']['obs'], skip_gen0=True))
            sets['SeqFuzz'] = cls.get_crash_grid_set(cls.parse_seqfuzz(cls.FILE_CONFIG['SeqFuzz']['obs'], cls.FILE_CONFIG['SeqFuzz']['log']))
        except Exception as e:
            print(f"[Error] MountainCar Processing: {e}")
            import traceback
            traceback.print_exc()
        
        # 简单检查数据是否为空
        if not any(len(v) > 0 for v in sets.values()):
            print("  [Warn] No MountainCar data found. Check filenames in 'mc/' folder.")
        else:
            print(f"  MC Data counts: { {k: len(v) for k, v in sets.items()} }")
        
        return sets

# ================= 2. BipedalWalker 数据处理模块 (bw/) =================

class BipedalWalkerLoader:
    BASE_DIR = "bw"
    FILES_CONFIG = {
        "CureFuzz": {"filename": "selection_log.pkl", "type": "pickle_curefuzz"},
        "MDPFuzz": {"filename": "fuzzer_10_0.01_0.01_0_logs.txt", "type": "csv_mdpfuzz"},
        "QDFuzz": {"filename": "1768120702.3916006_data.csv", "type": "csv_qdfuzz"},
        "SeqFuzz": {"filename": "all_run_seeds_0.pkl", "type": "pickle_seqfuzz"}
    }

    @staticmethod
    def normalize_key(key_raw):
        if isinstance(key_raw, bytes): return str(key_raw)
        if isinstance(key_raw, (list, tuple, np.ndarray)): return str(tuple(key_raw))
        return str(key_raw)

    @classmethod
    def get_data(cls):
        print("Processing BipedalWalker (bw/)...")
        sets = {}
        for name, config in cls.FILES_CONFIG.items():
            path = os.path.join(cls.BASE_DIR, config['filename'])
            ftype = config['type']
            unique_crashes = set()
            if not os.path.exists(path):
                print(f"  [Warn] BW File missing: {path}")
                sets[name] = set()
                continue
            try:
                if ftype == 'pickle_curefuzz':
                    with open(path, 'rb') as f:
                        for entry in pickle.load(f):
                            if entry.get('did_crash', False):
                                unique_crashes.add(cls.normalize_key(entry.get('mutate_state')))
                elif ftype == 'pickle_seqfuzz':
                    with open(path, 'rb') as f:
                        for entry in pickle.load(f):
                            if entry.get('crashed', False):
                                unique_crashes.add(cls.normalize_key(entry.get('state')))
                elif ftype == 'csv_mdpfuzz':
                    df = pd.read_csv(path, delimiter=';', engine='python', on_bad_lines='skip', skipinitialspace=True)
                    if 'Oracle' in df.columns:
                         df['Oracle'] = df['Oracle'].astype(str).map({'True': True, 'False': False, 'None': False}).fillna(False)
                    gen_col = next((c for c in df.columns if c.lower() == 'generation'), None)
                    if gen_col: df = df[(df[gen_col] != 0)]
                    for row in df[df['Oracle'] == True].itertuples():
                        if getattr(row, 'Input', None): unique_crashes.add(cls.normalize_key(getattr(row, 'Input')))
                elif ftype == 'csv_qdfuzz':
                    df = pd.read_csv(path)
                    if 'is_faulty' in df.columns:
                        for row in df[df['is_faulty'] == True].itertuples():
                            if getattr(row, 'input', None): unique_crashes.add(cls.normalize_key(getattr(row, 'input')))
                sets[name] = unique_crashes
            except Exception as e:
                print(f"  [Error] BW {name}: {e}")
                sets[name] = set()
        return sets

# ================= 3. CARLA 数据处理模块 (carla/) =================

class CarlaLoader:
    BASE_DIR = "carla"
    FILE_MAP = {"curefuzz.csv": "CureFuzz", "mdpfuzz.csv": "MDPFuzz", "qdfuzz.csv": "QDFuzz", "seqfuzz.csv": "SeqFuzz"}
    MAX_SEEDS = 1000
    TARGET_PHASE = "Phase2"

    @classmethod
    def get_data(cls):
        print("Processing CARLA (carla/)...")
        sets = {}
        for fname, label in cls.FILE_MAP.items():
            path = os.path.join(cls.BASE_DIR, fname)
            if not os.path.exists(path):
                print(f"  [Warn] CARLA File missing: {path}")
                sets[label] = set()
                continue
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                if 'phase' in df.columns and cls.TARGET_PHASE in df['phase'].values:
                    df = df[df['phase'] == cls.TARGET_PHASE]
                if len(df) > cls.MAX_SEEDS: df = df.iloc[:cls.MAX_SEEDS]
                unique_seeds = set()
                is_crash = None
                if 'success' in df.columns: is_crash = df['success'] == False
                elif 'collision' in df.columns:
                     is_crash = df['collision'].astype(str).str.lower() == 'true' if df['collision'].dtype == object else df['collision'] == True
                if is_crash is not None:
                    crashes = df[is_crash]
                    w_col = 'weather_id' if 'weather_id' in crashes.columns else 'weather'
                    if 'start_id' in crashes.columns:
                        for _, row in crashes.iterrows():
                            unique_seeds.add((row.get(w_col, -1), row['start_id'], row['target_id']))
                sets[label] = unique_seeds
            except Exception as e:
                print(f"  [Error] CARLA {label}: {e}")
                sets[label] = set()
        return sets

# ================= 主程序：合并绘图 =================

def main():
    # 1. 加载所有数据
    mc_data = MountainCarLoader.get_data()
    bw_data = BipedalWalkerLoader.get_data()
    carla_data = CarlaLoader.get_data()

    # 2. 准备画布 (1行3列)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    # 调整整体布局
    plt.subplots_adjust(top=0.82, wspace=0.3, bottom=0.15)

    # 3. 确定图例标签顺序
    all_keys = set(mc_data.keys()) | set(bw_data.keys()) | set(carla_data.keys())
    sorted_keys = sorted(list(all_keys)) 
    print(f"Keys for plotting: {sorted_keys}")

    def align_data(data_dict):
        """确保传入 venn 的字典包含所有 key，缺失的用空集合补齐"""
        return {k: data_dict.get(k, set()) for k in sorted_keys}

    # 4. 绘制三个子图
    
    # --- Subplot 1: MountainCar ---
    if any(len(v) > 0 for v in mc_data.values()):
        venn(align_data(mc_data), cmap=CMAP_NAME, alpha=ALPHA_VAL, legend_loc=None, ax=axes[0], fontsize=10)
    else:
        axes[0].text(0.5, 0.5, "No Data (mc/)", ha='center', va='center', fontsize=12)
        axes[0].axis('off')
    axes[0].set_title("MountainCar", fontsize=14, y=-0.15)

    # --- Subplot 2: BipedalWalker ---
    if any(len(v) > 0 for v in bw_data.values()):
        venn(align_data(bw_data), cmap=CMAP_NAME, alpha=ALPHA_VAL, legend_loc=None, ax=axes[1], fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "No Data (bw/)", ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    axes[1].set_title("BipedalWalker", fontsize=14, y=-0.15)

    # --- Subplot 3: CARLA ---
    if any(len(v) > 0 for v in carla_data.values()):
        venn(align_data(carla_data), cmap=CMAP_NAME, alpha=ALPHA_VAL, legend_loc=None, ax=axes[2], fontsize=10)
    else:
        axes[2].text(0.5, 0.5, "No Data (carla/)", ha='center', va='center', fontsize=12)
        axes[2].axis('off')
    axes[2].set_title("CARLA", fontsize=14, y=-0.15)

    # 5. 创建全局图例
    cmap = plt.get_cmap(CMAP_NAME)
    n_groups = len(sorted_keys)
    if n_groups > 0:
        # 严格使用 numpy.linspace 确保颜色匹配
        colors = [cmap(x) for x in np.linspace(0, 1, n_groups)]
        handles = [mpatches.Patch(color=colors[i], label=sorted_keys[i], alpha=ALPHA_VAL) for i in range(n_groups)]
        
        fig.legend(handles=handles, loc='upper center', ncol=n_groups, 
                   bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=13)

    # 6. 保存与显示
    print(f"Saving to {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE, dpi=300) 
    print("Done.")

if __name__ == "__main__":
    main()