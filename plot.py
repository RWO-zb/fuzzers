import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns

# ==========================================
# 1. 全局配置 (保持您的路径配置不变)
# ==========================================

# 预定义的样式和颜色
sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)
COLORS = {
    'Random':   '#7f8c8d',  # 灰色
    'MDPFuzz':  '#e74c3c',  # 红色
    'SeqFuzz':  '#2ecc71',  # 绿色
    'CureFuzz': '#9b59b6',  # 紫色
    'QDFuzz':   '#f39c12',  # 橙色
    'G-Model':  '#3498db',  # 蓝色
}

# 方法配置：名称、文件路径、解析器类型、标签名称
CONFIG = [
    {
        'name': 'Random',
        'file_path': 'rt_10_0.01_0.01_0_logs.txt',  
        'parser': 'mdpfuzz',                
        'color': COLORS['Random']
    },
    {
        'name': 'MDPFuzz',
        'file_path': 'fuzzer_10_0.01_0.01_0_logs.txt', 
        'parser': 'mdpfuzz',
        'color': COLORS['MDPFuzz']
    },
    {
        'name': 'SeqFuzz',
        'file_path': 'all_run_seeds_0.pkl', 
        'parser': 'seqfuzz',
        'color': COLORS['SeqFuzz']
    },
    {
        'name': 'CureFuzz',
        'file_path': 'selection_log.pkl',   
        'parser': 'curefuzz',
        'color': COLORS['CureFuzz']
    },
    {
        'name': 'QDFuzz',
        'file_path': '1765639810.5339673_data.csv',     
        'parser': 'qdfuzz',
        'color': COLORS['QDFuzz']
    },
    {
        'name': 'G-Model',
        'file_path': 'all_test_cases_log.pkl', 
        'parser': 'gmodel',
        'color': COLORS['G-Model']
    }
]

OUTPUT_FILE = 'comparison_crashes_over_time.png'

# ==========================================
# 2. 各方法的特定解析器
# ==========================================

def parse_mdpfuzz_format(file_path):
    """
    解析 MDPFuzz 和 Random (csv with ';')
    """
    if not os.path.exists(file_path):
        print(f"[警告] 文件不存在: {file_path}")
        return None

    try:
        # 1. 加载
        df = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip', skipinitialspace=True)
        
        # 2. 预处理列
        if 'Oracle' in df.columns and df['Oracle'].dtype == 'object':
            df['Oracle'] = df['Oracle'].map({'True': True, 'False': False, 'None': None})
        
        # 假设 Oracle == True 是崩溃
        df['is_crash'] = (df['Oracle'] == True)
        
        # 确保 RunTime 是数值
        if 'RunTime' in df.columns:
            df['RunTime'] = pd.to_numeric(df['RunTime'], errors='coerce')
        else:
            print(f"[错误] {file_path} 中缺少 'RunTime' 列")
            return None

        # 3. 去重 (基于 Input) - 保留首次出现
        if 'Input' not in df.columns:
            print(f"[错误] {file_path} 中缺少 'Input' 列")
            return None
            
        unique_df = df.drop_duplicates(subset=['Input'], keep='first')
        
        # 4. 提取崩溃时间
        crash_df = unique_df[unique_df['is_crash'] == True].copy()
        if crash_df.empty:
            return []

        # 5. 时间标准化 (绝对时间戳 -> 相对秒数)
        # 使用原始数据中最早的时间作为开始时间
        start_time = df['RunTime'].min()
        crash_times = crash_df['RunTime'] - start_time
        
        # 排序
        crash_times = sorted(crash_times.tolist())
        return crash_times

    except Exception as e:
        print(f"[错误] 解析 MDPFuzz 格式文件 {file_path} 时出错: {e}")
        return None


def parse_seqfuzz_format(file_path):
    """
    解析 SeqFuzz (pickle)
    """
    if not os.path.exists(file_path):
        print(f"[警告] 文件不存在: {file_path}")
        return None

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 去重逻辑: 基于 'state' bytes
        seen_states = set()
        dedup_crashes_timestamps = []

        # 动态检测 dtype 大小
        int32_size = 60
        int64_size = 120
        
        for entry in data:
            state = entry.get('state')
            if state is None: continue
            
            try:
                state_bytes = state.tobytes()
            except AttributeError:
                continue

            # 简单的长度检查
            if len(state_bytes) not in [int32_size, int64_size]:
                continue
                
            if state_bytes not in seen_states:
                seen_states.add(state_bytes)
                
                if entry.get('crashed', False):
                    t = entry.get('timestamp')
                    if t is not None:
                        dedup_crashes_timestamps.append(t)
        
        # 排序
        dedup_crashes_timestamps.sort()
        # 处理时间戳 (如果是 epoch 则归一化)
        if dedup_crashes_timestamps and dedup_crashes_timestamps[0] > 1e9:
             start_time = min(dedup_crashes_timestamps)
             dedup_crashes_timestamps = [t - start_time for t in dedup_crashes_timestamps]

        return dedup_crashes_timestamps

    except Exception as e:
        print(f"[错误] 解析 SeqFuzz 文件 {file_path} 时出错: {e}")
        return None


def parse_curefuzz_format(file_path):
    """
    解析 CureFuzz (pickle)
    """
    if not os.path.exists(file_path):
        print(f"[警告] 文件不存在: {file_path}")
        return None
        
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        seen_states = set()
        crash_times = []
        
        int32_size = 60
        int64_size = 120

        for entry in data:
            state = entry.get('mutate_state')
            if state is None: continue
            
            try:
                state_bytes = state.tobytes()
            except AttributeError:
                continue
            
            if len(state_bytes) not in [int32_size, int64_size]:
                continue
                
            if state_bytes not in seen_states:
                seen_states.add(state_bytes)
                
                if entry.get('did_crash', False):
                    t = entry.get('elapsed_time')
                    if t is not None:
                        crash_times.append(t)
        
        crash_times.sort()
        return crash_times 

    except Exception as e:
        print(f"[错误] 解析 CureFuzz 文件 {file_path} 时出错: {e}")
        return None


def parse_qdfuzz_format(file_path):
    """
    解析 QDFuzz (csv)
    """
    if not os.path.exists(file_path):
        print(f"[警告] 文件不存在: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # 必需列检查
        if 'input' not in df.columns or 'is_faulty' not in df.columns or 'elapsed_time' not in df.columns:
            print(f"[错误] QDFuzz 文件缺少必要的列")
            return None
            
        # 转换 is_faulty
        if df['is_faulty'].dtype == 'object':
             df['is_faulty'] = df['is_faulty'].astype(str).str.lower() == 'true'
        else:
             df['is_faulty'] = df['is_faulty'].astype(bool)
             
        # 确保 elapsed_time 是数值
        df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce').fillna(0.0)

        # 去重: 基于 'input'，保留首次出现
        unique_df = df.dropna(subset=['input']).drop_duplicates(subset=['input'], keep='first')
        
        # 提取故障
        crash_df = unique_df[unique_df['is_faulty'] == True]
        
        crash_times = sorted(crash_df['elapsed_time'].tolist())
        return crash_times

    except Exception as e:
        print(f"[错误] 解析 QDFuzz 文件 {file_path} 时出错: {e}")
        return None


def parse_gmodel_format(file_path):
    """
    解析 G-Model (pickle)
    【修改说明】: 此函数现在会统计日志中 *所有* 的崩溃 (Random + Generative)，
    不再按 source='generative' 进行过滤。
    """
    if not os.path.exists(file_path):
        print(f"[警告] 文件不存在: {file_path}")
        return None
        
    try:
        with open(file_path, 'rb') as f:
            log_data = pickle.load(f)

        unique_crashes_timestamps = {}
        
        for entry in log_data:
            # 移除了 source 过滤，统计所有来源的 crash
            # if entry.get('source') != 'generative': continue 

            if entry.get('is_crash'):
                # 使用 tuple(input) 对所有 crash 进行统一去重
                t_in = tuple(entry['input'])
                timestamp = entry.get('time', 0)
                
                # 如果这个输入之前没出现过（或者想要保留最早的一次），加入字典
                if t_in not in unique_crashes_timestamps:
                    unique_crashes_timestamps[t_in] = timestamp
        
        crash_times = sorted(list(unique_crashes_timestamps.values()))
        return crash_times # time 字段看起来已经是 elapsed time (秒)

    except Exception as e:
        print(f"[错误] 解析 G-Model 文件 {file_path} 时出错: {e}")
        return None


# ==========================================
# 3. 绘图主逻辑
# ==========================================

def plot_comparison(config_list):
    plt.figure(figsize=(12, 8))
    
    plotted_count = 0
    
    for cfg in config_list:
        name = cfg['name']
        fpath = cfg['file_path']
        parser_type = cfg['parser']
        color = cfg['color']
        
        print(f"正在处理 {name} ({parser_type})...")
        
        crash_times = None
        
        # 调用对应的解析器
        if parser_type == 'mdpfuzz':
            crash_times = parse_mdpfuzz_format(fpath)
        elif parser_type == 'seqfuzz':
            crash_times = parse_seqfuzz_format(fpath)
        elif parser_type == 'curefuzz':
            crash_times = parse_curefuzz_format(fpath)
        elif parser_type == 'qdfuzz':
            crash_times = parse_qdfuzz_format(fpath)
        elif parser_type == 'gmodel':
            crash_times = parse_gmodel_format(fpath)
        
        if crash_times is None or len(crash_times) == 0:
            print(f"  -> 无数据或文件缺失，跳过 {name}。")
            continue
            
        # 转换为小时
        times_hours = [t / 3600.0 for t in crash_times]
        
        # 构造累积计数 (1, 2, 3...)
        cumulative_counts = list(range(1, len(times_hours) + 1))
        
        # 添加起点 (0, 0)
        if times_hours[0] > 0:
            times_hours.insert(0, 0)
            cumulative_counts.insert(0, 0)
            
        # 绘制阶梯图
        plt.step(times_hours, cumulative_counts, where='post', 
                 label=f'{name} ({cumulative_counts[-1]})', 
                 color=color, linewidth=2.5)
        
        plotted_count += 1
        print(f"  -> {name}: 最终发现 {cumulative_counts[-1]} 个独特崩溃，耗时 {times_hours[-1]:.2f} 小时。")

    if plotted_count == 0:
        print("没有绘制任何数据。请检查文件路径配置。")
        return

    # 图表装饰
    plt.title('Comparison of Unique Crashes Found Over Time', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Time Elapsed (Hours)', fontsize=14)
    plt.ylabel('Cumulative Unique Crashes', fontsize=14)
    
    plt.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    sns.despine()
    plt.tight_layout()
    
    try:
        plt.savefig(OUTPUT_FILE, dpi=300)
        print(f"\n[成功] 对比图已保存到: {OUTPUT_FILE}")
    except Exception as e:
        print(f"[错误] 保存图表时失败: {e}")

if __name__ == "__main__":
    print("--- 多方法崩溃发现对比绘图脚本 ---")
    plot_comparison(CONFIG)