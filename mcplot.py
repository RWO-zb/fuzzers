import os
import pickle
import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ==========================================
# 1. 配置区域 (请根据实际文件路径修改)
# ==========================================
CONFIG = {
    # SeqFuzz (pickle)
    'SeqFuzz': {
        'path': 'all_run_seeds_0.pkl',
        'enabled': True,
        'color': '#9C27B0',  # 紫色
        'format': 'seqfuzz'
    },
    # MdpFuzz (csv/txt, sep=';')
    'MdpFuzz': {
        'path': 'MC_DQN_NoCov_5_0.01_0.1_0_12h_logs.txt', # 请修改为您的 MdpFuzz 日志文件名
        'enabled': True,
        'color': '#E64A19',  # 深橙色
        'format': 'mdpfuzz'
    },
    # Random (假设格式同 MdpFuzz)
    'Random': {
        'path': 'MC_DQN_RT_0_budget730000_logs.txt',  # 请修改为您的 Random 日志文件名
        'enabled': True,
        'color': '#757575',  # 灰色
        'format': 'mdpfuzz' # 假设 Random 使用与 MdpFuzz 相同的日志格式
    },
    # CureFuzz (pickle)
    'CureFuzz': {
        'path': 'selection_log.pkl',
        'enabled': True,
        'color': '#2196F3',  # 蓝色
        'format': 'curefuzz'
    },
    # QDFuzz (csv)
    'QDFuzz': {
        'path': 'mc_test_data.csv',
        'enabled': True,
        'color': '#FFC107',  # 琥珀色
        'format': 'qdfuzz'
    },
    # G-Model (pickle)
    'G-Model': {
        'path': 'all_test_cases_log.pkl',
        'enabled': True,
        'color': '#009688',  # 青色
        'format': 'gmodel'
    }
}

OUTPUT_FILENAME = 'compare_crash_discovery_over_time.png'
sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)

# ==========================================
# 2. 数据加载与处理函数
# ==========================================

def load_pickle(filepath):
    if not os.path.exists(filepath):
        print(f"[跳过] 文件未找到: {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[错误] 加载 pickle {filepath} 失败: {e}")
        return None

def process_seqfuzz(filepath):
    """处理 SeqFuzz 数据"""
    data = load_pickle(filepath)
    if not data: return []
    
    # 去重逻辑 (参考 seqfuzzplot.py)
    seen_states = set()
    crash_times = []
    
    for entry in data:
        # 获取状态
        state = entry.get('state')
        if state is None: state = entry.get('mutate_state')
        if state is None: continue
        
        try:
            state_bytes = state.tobytes() if hasattr(state, 'tobytes') else bytes(state)
        except:
            continue
            
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            # 检查是否崩溃
            if entry.get('crashed') or entry.get('did_crash', False):
                t = entry.get('crash_time')
                if t is not None:
                    crash_times.append(t)
    
    return sorted(crash_times)

def process_mdpfuzz_style(filepath):
    """处理 MdpFuzz/Random 数据 (CSV格式)"""
    if not os.path.exists(filepath):
        print(f"[跳过] 文件未找到: {filepath}")
        return []
    
    try:
        # 参考 mdpfuzzplot.py
        df = pd.read_csv(filepath, sep=';', skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        
        # 过滤掉前 10000 个初始样本 (参考 mdpfuzzplot.py logic)
        SKIP_INITIAL = 10000
        if len(df) > SKIP_INITIAL:
            df = df.iloc[SKIP_INITIAL:].copy()
            
        df['Oracle'] = df['Oracle'].astype(str).str.strip() == 'True'
        
        # 提取崩溃
        crashes = df[df['Oracle'] == True].copy()
        if crashes.empty: return []
        
        # 处理时间 (优先使用 CrashTime, 如果没有则尝试计算)
        if 'CrashTime' not in crashes.columns:
            crashes['CrashTime'] = np.nan
        
        if 'RunTime' in crashes.columns:
            crashes['RunTime'] = pd.to_numeric(crashes['RunTime'], errors='coerce')
            start_time = df['RunTime'].min()
            # 填充缺失的 CrashTime
            mask = crashes['CrashTime'].isna()
            crashes.loc[mask, 'CrashTime'] = crashes.loc[mask, 'RunTime'] - start_time
            
        # 基于 Input 去重
        unique_crashes = crashes.drop_duplicates(subset=['Input'])
        
        crash_times = unique_crashes['CrashTime'].dropna().values.tolist()
        return sorted(crash_times)
        
    except Exception as e:
        print(f"[错误] 读取 CSV {filepath} 失败: {e}")
        return []

def process_curefuzz(filepath):
    """处理 CureFuzz 数据"""
    data = load_pickle(filepath)
    if not data: return []
    
    seen_states = set()
    crash_times = []
    
    for entry in data:
        state = entry.get('mutate_state')
        if state is None: continue
        
        state_bytes = state.tobytes()
        
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            if entry.get('did_crash', False):
                t = entry.get('crash_time')
                if t is not None:
                    crash_times.append(t)
                    
    return sorted(crash_times)

def process_qdfuzz(filepath):
    """处理 QDFuzz 数据"""
    if not os.path.exists(filepath):
        print(f"[跳过] 文件未找到: {filepath}")
        return []
    
    try:
        df = pd.read_csv(filepath)
        crash_times = []
        seen_states = set()
        
        for _, row in df.iterrows():
            # 检查是否崩溃
            if not row.get('is_faulty', False):
                continue
                
            # 去重 (基于 input 解析出的 state)
            inp = row.get('input')
            if isinstance(inp, str):
                try:
                    inp = ast.literal_eval(inp)
                except:
                    pass
            
            state = np.array(inp, dtype=np.float32) if isinstance(inp, list) else inp
            if state is None: continue
            
            try:
                state_bytes = state.tobytes()
                if state_bytes not in seen_states:
                    seen_states.add(state_bytes)
                    t = row.get('discovery_time')
                    if t is not None:
                        crash_times.append(t)
            except:
                continue
                
        return sorted(crash_times)
    except Exception as e:
        print(f"[错误] 读取 QDFuzz CSV 失败: {e}")
        return []

def process_gmodel(filepath):
    """处理 G-Model 数据"""
    data = load_pickle(filepath)
    if not data: return []
    
    seen_inputs = set()
    crash_times = []
    
    for entry in data:
        # 仅统计 generative 来源 (如果需要包含所有来源，请注释掉下行)
        # if entry.get('source') != 'generative': continue
        
        if entry.get('is_crash'):
            t_in = tuple(entry['input'])
            if t_in not in seen_inputs:
                seen_inputs.add(t_in)
                t = entry.get('timestamp', 0)
                crash_times.append(t)
                
    return sorted(crash_times)

# ==========================================
# 3. 主绘图函数
# ==========================================

def plot_combined_crashes():
    plt.figure(figsize=(12, 8))
    
    max_time_hours = 0
    has_data = False
    
    print("开始处理数据...")
    
    for label, config in CONFIG.items():
        if not config['enabled']:
            continue
            
        fmt = config['format']
        path = config['path']
        
        times = []
        if fmt == 'seqfuzz':
            times = process_seqfuzz(path)
        elif fmt == 'mdpfuzz':
            times = process_mdpfuzz_style(path)
        elif fmt == 'curefuzz':
            times = process_curefuzz(path)
        elif fmt == 'qdfuzz':
            times = process_qdfuzz(path)
        elif fmt == 'gmodel':
            times = process_gmodel(path)
            
        if not times:
            print(f"  - {label}: 无数据或加载失败")
            continue
            
        print(f"  - {label}: 加载了 {len(times)} 个唯一崩溃")
        has_data = True
        
        # 转换为小时
        times_hours = [t / 3600.0 for t in times]
        counts = list(range(1, len(times) + 1))
        
        # 记录最大时间用于设置坐标轴
        if times_hours:
            max_time_hours = max(max_time_hours, max(times_hours))
        
        # 绘制曲线
        plt.plot(times_hours, counts, 
                 label=label, 
                 color=config['color'], 
                 linewidth=2.5, 
                 alpha=0.85)
        
        # 绘制终点标记
        if times_hours:
            plt.scatter(times_hours[-1], counts[-1], color=config['color'], s=40)

    if not has_data:
        print("所有启用的方法均无有效数据，无法绘图。")
        return

    # 设置图表样式
    plt.title('Unique Crash Discovery Over Time (Method Comparison)', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Time (Hours)', fontsize=14, labelpad=10)
    plt.ylabel('Cumulative Unique Crashes', fontsize=14, labelpad=10)
    
    plt.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    sns.despine()
    plt.tight_layout()
    
    try:
        plt.savefig(OUTPUT_FILENAME, dpi=300)
        print(f"\n图表已保存至: {OUTPUT_FILENAME}")
    except Exception as e:
        print(f"保存图表失败: {e}")
    plt.close()

if __name__ == "__main__":
    plot_combined_crashes()