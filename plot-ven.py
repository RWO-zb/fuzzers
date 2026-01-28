import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from venn import venn

# ================= 配置区域 =================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

OUTPUT_FILE = "Venn.pdf"
CMAP_NAME = "plasma"
ALPHA_VAL = 0.3

# ================= 辅助函数 =================

def extract_root_id(entry):
    """
    从字典或行中提取 Root ID。
    """
    keys = ['root_seed', 'root_id', 'RootID', 'parent_id', 'seed_id', 'id', 'SeedID']
    
    def get_val(container, key):
        if isinstance(container, dict):
            return container.get(key)
        elif isinstance(container, pd.Series):
            return container[key] if key in container else None
        return None

    for k in keys:
        val = get_val(entry, k)
        if val is None: continue
        
        # 跳过向量类型的 ID
        if isinstance(val, (np.ndarray, list)):
            continue 
            
        # 处理标量 ID
        try:
            if pd.notna(val) and str(val).lower() != 'none':
                try:
                    return str(int(float(val)))
                except:
                    return str(val)
        except (ValueError, TypeError):
            return str(val)
            
    return None

def normalize_input_key(val, precision=4):
    """
    标准化 Input/State 向量。
    """
    try:
        # 1. 处理字符串类型的列表 "[0.1, 0.2]"
        if isinstance(val, str):
            val = val.strip()
            val = val.replace('[', '').replace(']', '').replace('\n', ' ')
            if ',' in val:
                val = [float(x) for x in val.split(',') if x.strip()]
            else:
                val = [float(x) for x in val.split() if x.strip()]
        
        # 2. 转为 numpy 数组
        arr = np.array(val, dtype=float).flatten()
        
        # 3. 返回 Tuple 以便哈希
        # 注意：这里的 precision 参数决定了保留的小数位数
        return tuple(np.round(arr, precision))
    except:
        return None

# ================= 1. MountainCar 数据处理 (Modified) =================

class MountainCarLoader:
    BASE_DIR = "mc"
    
    FILE_CONFIG = {
        "CureFuzz": {
            "crash_file": "cure_crash.pkl",
        },
        "MDPFuzz": {
            "obs_file": "MC_DQN_NoCov_5_0.01_0.1_0_7000it_obs.txt",
            "log_file": "MC_DQN_NoCov_5_0.01_0.1_0_7000it_logs.txt"
        },
        "QDFuzz": {
            "csv_file": "mc_test_data.csv"
        },
        "SeqDivFuzz": {
            "log_file": "all_run_seeds_0.pkl"
        }
    }

    @staticmethod
    def _get_path(filename):
        return os.path.join(MountainCarLoader.BASE_DIR, os.path.basename(filename))

    @staticmethod
    def _normalize_mc_input(val):
        # [修改] 根据您的要求，这里将精度设为 8，以匹配您的8位小数输入，避免截断
        # 仅保留前2维 (pos, vel)
        res = normalize_input_key(val, precision=8)
        if res and len(res) > 2:
            return res[:2]
        return res

    @classmethod
    def _build_mdp_obs_map(cls, obs_path):
        """
        MDPFuzz 辅助: 建立 SeedID -> Input Vector 的映射
        [修改] 增强 JSON 解析逻辑，参考 mdpfuzz-RQ2.py
        """
        id_map = {} 
        if not os.path.exists(obs_path):
            return id_map

        print(f"    [MC MDP] Parsing Obs for Init Seeds: {os.path.basename(obs_path)}")
        try:
            with open(obs_path, 'r') as f:
                current_info = None
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    if line.startswith("--- Test Case Info:"):
                        try:
                            # 参考 mdpfuzz-RQ2.py 的切片方式，更稳健
                            # json_part = line[len("--- Test Case Info: "):-len(" ---")]
                            # 也可以用更通用的 replace 防止格式微小差异
                            json_str = line.replace("--- Test Case Info:", "").replace("---", "").strip()
                            current_info = json.loads(json_str)
                        except:
                            current_info = None
                    else:
                        if current_info:
                            # 只有 Gen 0 才是初始种子
                            # mdpfuzz-RQ2.py 中通过判断 gen==0 来 skip，说明初始种子在 Gen 0
                            gen = current_info.get('Generation', -1)
                            if gen == 0:
                                sid = extract_root_id(current_info)
                                # 如果 JSON 中没有直接的 ID，尝试用 Episode 序号等(视具体数据而定)
                                # 但这里我们严格依赖 ID 以匹配 Log
                                if sid and sid not in id_map:
                                    # 读取该 Episode 的第一行数据作为初始输入
                                    val = cls._normalize_mc_input(line)
                                    if val:
                                        id_map[sid] = val
                            current_info = None # 只读第一帧
        except Exception as e:
            print(f"    [Error] parsing MDP obs: {e}")
        return id_map

    @classmethod
    def process_curefuzz(cls, cfg):
        path = cls._get_path(cfg["crash_file"])
        seeds = set()
        inputs = set()
        
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                
                for entry in data:
                    if isinstance(entry, dict):
                        rid = entry.get('root_id')
                        if rid is None: rid = extract_root_id(entry)
                        
                        if rid is not None:
                            # 逻辑：One Input per Unique Seed
                            if rid not in seeds:
                                seeds.add(rid)
                                rseed = entry.get('root_seed')
                                if rseed is None:
                                    obs = entry.get('obs_seq') or entry.get('state')
                                    if obs is not None and len(obs) > 0:
                                        rseed = obs[0] if isinstance(obs[0], (list, np.ndarray)) else obs
                                
                                if rseed is not None:
                                    norm = cls._normalize_mc_input(rseed)
                                    if norm: inputs.add(norm)
            except Exception as e:
                print(f"    [Error] MC CureFuzz: {e}")
        
        return inputs, len(seeds)

    @classmethod
    def process_mdpfuzz(cls, cfg):
        # 1. 建立 ID 映射 (使用高精度)
        obs_path = cls._get_path(cfg["obs_file"])
        id_map = cls._build_mdp_obs_map(obs_path)
        
        # 2. 读取 Log 获取 Crash SeedID
        log_path = cls._get_path(cfg["log_file"])
        seeds = set()
        inputs = set()
        
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    header = f.readline().strip().split('; ')
                    if 'SeedID' in header and 'Oracle' in header:
                        idx_sid = header.index('SeedID')
                        idx_oracle = header.index('Oracle')
                        
                        for line in f:
                            parts = line.strip().split('; ')
                            if len(parts) > max(idx_sid, idx_oracle):
                                oracle = parts[idx_oracle]
                                sid = parts[idx_sid]
                                
                                # Strict Logic: Oracle==True and SeedID!=None
                                if oracle == 'True' and sid != 'None':
                                    if sid not in seeds:
                                        seeds.add(sid)
                                        if sid in id_map:
                                            inputs.add(id_map[sid])
            except Exception as e:
                print(f"    [Error] MC MDPFuzz: {e}")
        
        return inputs, len(seeds)

    @classmethod
    def process_qdfuzz(cls, cfg):
        csv_path = cls._get_path(cfg["csv_file"])
        seeds = set()
        inputs = set()
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                col_fault = next((c for c in ['is_faulty', 'crashed'] if c in df.columns), None)
                col_seed = 'seed_id' 
                
                if col_fault and col_seed in df.columns:
                    crashes = df[df[col_fault] == True]
                    for _, row in crashes.iterrows():
                        sid = row[col_seed]
                        if pd.notna(sid) and str(sid) != '-1':
                            if sid not in seeds:
                                seeds.add(sid)
                                val = row.get('input')
                                if val is not None:
                                    norm = cls._normalize_mc_input(val)
                                    if norm: inputs.add(norm)
            except Exception as e:
                print(f"    [Error] MC QDFuzz: {e}")
            
        return inputs, len(seeds)

    @classmethod
    def process_seqdivfuzz(cls, cfg):
        path = cls._get_path(cfg["log_file"])
        seeds = set()
        inputs = set()
        
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                
                for entry in data:
                    if entry.get('crashed', False):
                        vec = entry.get('root_seed')
                        if vec is not None:
                            # 这里的 norm 现在是保留8位小数
                            norm = cls._normalize_mc_input(vec)
                            if norm:
                                if norm not in seeds:
                                    seeds.add(norm)
                                    inputs.add(norm)
            except Exception as e:
                print(f"    [Error] MC SeqDivFuzz: {e}")
                
        return inputs, len(seeds)

    @classmethod
    def get_data(cls):
        results = {}
        for name in ["CureFuzz", "MDPFuzz", "QDFuzz", "SeqDivFuzz"]:
            func = getattr(cls, f"process_{name.lower()}")
            inputs, count = func(cls.FILE_CONFIG[name])
            results[name] = (inputs, count)
            print(f"  MC {name}: {count} unique seeds -> {len(inputs)} representative inputs (Precision=8).")
        return results

# ================= 2. BipedalWalker 数据处理 (Keep Original Precision=2) =================

class BipedalWalkerLoader:
    BASE_DIR = "bw"
    FILES_CONFIG = {
        "CureFuzz": {"filename": "selection_log.pkl", "type": "pickle"},
        "MDPFuzz": {"filename": "fuzzer_10_0.01_0.01_0_logs.txt", "type": "csv_mdpfuzz"},
        "QDFuzz": {"filename": "1769257333.763425_data.csv", "type": "csv_qdfuzz"},
        "SeqDivFuzz": {"filename": "all_run_seeds_0.pkl", "type": "pickle"}
    }

    @classmethod
    def get_data(cls):
        print("Processing BipedalWalker (One Input per Unique Seed)...")
        results = {}
        
        for name, config in cls.FILES_CONFIG.items():
            path = os.path.join(cls.BASE_DIR, config['filename'])
            inputs = set()
            seeds = set()
            
            if not os.path.exists(path):
                print(f"  [Warning] BW file not found: {path}")
                results[name] = (set(), 0)
                continue
                
            try:
                # --- Pickle Log ---
                if config['type'] == 'pickle':
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                        for i, entry in enumerate(data):
                            if entry.get('crashed', False) or entry.get('did_crash', False):
                                root_seed = entry.get('root_seed')
                                rid = None
                                if root_seed is not None:
                                    if isinstance(root_seed, np.ndarray): rid = root_seed.tobytes()
                                    elif hasattr(root_seed, 'tobytes'): rid = root_seed.tobytes()
                                    else: rid = np.array(root_seed).tobytes()
                                if rid is None: rid = extract_root_id(entry)
                                if rid is None: rid = f"unknown_{i}"

                                if rid not in seeds:
                                    seeds.add(rid)
                                    state = entry.get('state')
                                    if state is None: state = entry.get('mutate_state')
                                    if state is not None:
                                        # Strict: BW keeps precision=2
                                        inputs.add(normalize_input_key(state, precision=2))

                # --- MDPFuzz CSV ---
                elif config['type'] == 'csv_mdpfuzz':
                    df = pd.read_csv(path, delimiter=';', engine='python', on_bad_lines='skip', skipinitialspace=True)
                    if 'Oracle' in df.columns:
                         df['Oracle'] = df['Oracle'].astype(str).map({'True': True, 'False': False, 'None': False}).fillna(False)
                    
                    crash_df = df[df['Oracle'] == True]
                    for _, row in crash_df.iterrows():
                        rid = extract_root_id(row)
                        if rid:
                            if rid not in seeds:
                                seeds.add(rid)
                                val = row.get('Input')
                                if pd.notna(val):
                                    inputs.add(normalize_input_key(val, precision=2))

                # --- QDFuzz CSV ---
                elif config['type'] == 'csv_qdfuzz':
                    df = pd.read_csv(path)
                    crash_col = next((c for c in ['is_faulty', 'crashed'] if c in df.columns), None)
                    if crash_col:
                        is_crash = df[crash_col].astype(str).str.lower() == 'true' if df[crash_col].dtype == object else df[crash_col] == True
                        crash_df = df[is_crash]
                        if 'mutation_count' in df.columns:
                            crash_df = crash_df[crash_df['mutation_count'] > 0]
                        for _, row in crash_df.iterrows():
                            rid = extract_root_id(row)
                            if rid and str(rid) != '-1':
                                if rid not in seeds:
                                    seeds.add(rid)
                                    val = row.get('input')
                                    if val is None: val = row.get('state')
                                    if pd.notna(val):
                                        inputs.add(normalize_input_key(val, precision=2))

                print(f"  BW {name}: {len(seeds)} unique seeds -> {len(inputs)} representative inputs.")
                results[name] = (inputs, len(seeds))
                
            except Exception as e:
                print(f"  [Error] BW {name}: {e}")
                results[name] = (set(), 0)
        return results

# ================= 3. CARLA 数据处理 (Unchanged) =================

class CarlaLoader:
    BASE_DIR = "carla"
    FILE_MAP = {"curefuzz.csv": "CureFuzz", "mdpfuzz.csv": "MDPFuzz", "qdfuzz.csv": "QDFuzz", "seqfuzz.csv": "SeqDivFuzz"}
    MAX_SEEDS = 1000
    TARGET_PHASE = "Phase2"

    @classmethod
    def get_data(cls):
        print("Processing CARLA (Weather+Route)...")
        results = {}
        for fname, label in cls.FILE_MAP.items():
            path = os.path.join(cls.BASE_DIR, fname)
            scenarios = set()
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    df.columns = [c.strip() for c in df.columns]
                    if 'phase' in df.columns and cls.TARGET_PHASE in df['phase'].values:
                        df = df[df['phase'] == cls.TARGET_PHASE]
                    if len(df) > cls.MAX_SEEDS: df = df.iloc[:cls.MAX_SEEDS]
                    
                    is_crash = None
                    if 'success' in df.columns: is_crash = df['success'] == False
                    elif 'collision' in df.columns:
                         is_crash = df['collision'].astype(str).str.lower() == 'true' if df['collision'].dtype == object else df['collision'] == True
                    
                    if is_crash is not None:
                        crashes = df[is_crash]
                        w_col = 'weather_id' if 'weather_id' in crashes.columns else 'weather'
                        if 'start_id' in crashes.columns:
                            for _, row in crashes.iterrows():
                                val = (row.get(w_col, -1), row['start_id'], row['target_id'])
                                scenarios.add(val)
                except Exception as e:
                    print(f"  [Error] CARLA {label}: {e}")
            results[label] = (scenarios, len(scenarios))
        return results

# ================= 主程序 =================

def main():
    print("=== Processing MountainCar ===")
    mc_data = MountainCarLoader.get_data()
    
    print("\n=== Processing BipedalWalker ===")
    bw_data = BipedalWalkerLoader.get_data()
    
    print("\n=== Processing CARLA ===")
    carla_data = CarlaLoader.get_data()

    # [修改建议 1] 进一步减小画布高度到 4.0 (更扁)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.0))
    
    # [修改建议 2] 进一步增大 top 到 0.92 (让图表上沿更靠近画布顶部)
    plt.subplots_adjust(top=0.92, wspace=0.3, bottom=0.15)
    
    all_keys = sorted(list(set(mc_data.keys()) | set(bw_data.keys()) | set(carla_data.keys())))

    def plot_hybrid_venn(ax, data_dict, title):
        venn_data = {}
        for k in all_keys:
            if k in data_dict:
                input_set, seed_count = data_dict[k]
                label = f"{k}\n(Seeds: {seed_count})"
                venn_data[label] = input_set
            else:
                venn_data[f"{k}\n(Seeds: 0)"] = set()
        
        # 移除空集以避免报错
        if not any(len(v) > 0 for v in venn_data.values()):
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.axis('off')
        else:
            # [修改] 显著增大 Venn 图内文字的大小以匹配 Draft 风格 (10 -> 20)
            venn(venn_data, cmap=CMAP_NAME, alpha=ALPHA_VAL, legend_loc=None, ax=ax, fontsize=14)
        
        # [修改] 增大标题文字大小 (14 -> 20)
        ax.set_title(title, fontsize=20, y=-0.15)

    plot_hybrid_venn(axes[0], mc_data, "(a) MountainCar")
    plot_hybrid_venn(axes[1], bw_data, "(b) BipedalWalker")
    plot_hybrid_venn(axes[2], carla_data, "(c) CARLA")

    # 生成图例
    cmap = plt.get_cmap(CMAP_NAME)
    n_groups = len(all_keys)
    if n_groups > 0:
        colors = [cmap(x) for x in np.linspace(0, 1, n_groups)]
        handles = [mpatches.Patch(color=colors[i], label=all_keys[i], alpha=ALPHA_VAL) for i in range(n_groups)]
        # [修改] 增大图例文字 (13 -> 16)，并微调 bbox_to_anchor 以适应大字体，保证与图表的紧凑间距
        fig.legend(handles=handles, loc='upper center', ncol=n_groups, 
                   bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=16)

    print(f"\nSaving to {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight') # 添加 bbox_inches='tight' 防止大字体被裁剪
    print("Done.")

if __name__ == "__main__":
    main()