import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator

# --- 1. 配置 ---
LOG_FILE = 'all_run_seeds_0.pkl'
OBS_FILE = 'all_episodes_obs.txt'

# 输出图片文件名
PLOT_1_NAME = '1_crash_discovery_over_time.png'
PLOT_2_NAME = '2_state_space_coverage.png'
PLOT_3_NAME = '3_mutation_depth_hist.png'
PLOT_4_NAME = '4_crashes_vs_unique_inputs.png'

# 设置 Seaborn 样式
sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)

# --- 2. 数据加载与处理 ---

def load_pickle(filepath):
    """加载 .pkl 日志文件"""
    if not os.path.exists(filepath):
        print(f"[错误] 未找到文件: {filepath}")
        return None
    try:
        print(f"正在加载日志: {filepath} ...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"成功加载 {len(data)} 条日志记录。")
        return data
    except Exception as e:
        print(f"[错误] 加载 pickle 失败: {e}")
        return None

def load_obs_txt(filepath):
    """
    加载 .txt 观测序列文件。
    格式说明:
    - 每行是一组坐标 (例如: -0.505, -0.0001,)
    - 不同剧集(episode)之间用 '######' 分隔
    """
    if not os.path.exists(filepath):
        print(f"[错误] 未找到文件: {filepath}")
        return None
    
    obs_data = []
    current_episode = []
    
    print(f"正在解析观测数据: {filepath} ...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                
                # 检查分隔符
                if '######' in line:
                    if current_episode:
                        obs_data.append(current_episode)
                        current_episode = []
                    continue
                
                # 解析坐标数据
                try:
                    # 去除可能的尾部逗号
                    if line.endswith(','):
                        line = line[:-1]
                    
                    parts = line.split(',')
                    # 转换为浮点数列表 [x, y]
                    point = [float(p.strip()) for p in parts if p.strip()]
                    
                    if point:
                        current_episode.append(point)
                except ValueError:
                    # 跳过无法解析的行
                    continue
        
        # 处理文件末尾，如果最后一个块没有分隔符
        if current_episode:
            obs_data.append(current_episode)
            
        print(f"成功加载 {len(obs_data)} 条观测序列。")
        return obs_data
        
    except Exception as e:
        print(f"[错误] 读取 txt 文件失败: {e}")
        return None

def deduplicate_data(selection_log, obs_sequences):
    """
    根据 state 去重日志，并同步过滤观测序列。
    确保 log 和 obs 是一一对应的。
    """
    print("正在去重数据...")
    seen_states = set()
    dedup_log = []
    dedup_obs = []
    
    # 确保长度一致，以较短者为准
    min_len = min(len(selection_log), len(obs_sequences))
    
    if len(selection_log) != len(obs_sequences):
        print(f"[警告] 日志长度 ({len(selection_log)}) 与 观测序列长度 ({len(obs_sequences)}) 不一致，将截断至 {min_len}。")

    for i in range(min_len):
        entry = selection_log[i]
        obs = obs_sequences[i]
        
        # --- 修复部分 START ---
        # 获取状态用于去重 (兼容 'state' 或 'mutate_state')
        # 注意：不能使用 `or` 运算符，因为如果 state 是 numpy 数组，
        # bool(array) 会抛出 "truth value ambiguous" 错误。
        state = entry.get('state')
        if state is None:
            state = entry.get('mutate_state')
        # --- 修复部分 END ---
        
        if state is None:
            continue
        
        # 转换为 bytes 以存入 set
        try:
            if hasattr(state, 'tobytes'):
                state_bytes = state.tobytes()
            else:
                # 假设已经是 bytes 或 tuple
                state_bytes = bytes(state)
        except:
            continue

        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            dedup_log.append(entry)
            dedup_obs.append(obs)
            
    print(f"去重完成。保留了 {len(dedup_log)} 条唯一记录。")
    return dedup_log, dedup_obs

# --- 3. 绘图函数 ---

def plot_1_crashes_over_time(selection_log, total_raw_count):
    """[图表1] 崩溃发现随时间的变化"""
    print(f"正在绘制: {PLOT_1_NAME}")
    
    crash_times = []
    for entry in selection_log:
        is_crash = entry.get('crashed') or entry.get('did_crash', False)
        if is_crash:
            t = entry.get('crash_time')
            if t is not None:
                crash_times.append(t)

    unique_crashes_count = len(crash_times)
    
    if not crash_times:
        print("[提示] 没有找到包含时间信息的崩溃数据，跳过图表 1。")
        return

    crash_times.sort()
    times_in_hours = [t / 3600.0 for t in crash_times]
    counts = range(1, len(crash_times) + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(times_in_hours, counts, color='#E64A19', linewidth=3, label='Unique Crashes')
    plt.fill_between(times_in_hours, counts, color='#E64A19', alpha=0.1)
    
    plt.title('Crash Discovery Over Time', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Time (Hours)', fontsize=14, labelpad=10)
    plt.ylabel('Cumulative Crashes', fontsize=14, labelpad=10)
    
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 统计信息框
    stats_text = (
        f"$\\bf{{Statistics}}$\n"
        f"Total Inputs: {total_raw_count}\n"
        f"Unique Inputs: {len(selection_log)}\n"
        f"Unique Crashes: {unique_crashes_count}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=13,
                   verticalalignment='top', horizontalalignment='left', bbox=props)
    
    sns.despine()
    plt.tight_layout()
    try:
        plt.savefig(PLOT_1_NAME, dpi=300)
        print(f"已保存: {PLOT_1_NAME}")
    except Exception as e:
        print(f"保存图表 1 失败: {e}")
    plt.close()

def plot_2_state_space(selection_log, obs_sequences):
    """[图表2] 状态空间覆盖图 (使用 txt 中的轨迹数据)"""
    print(f"正在绘制: {PLOT_2_NAME}")
    
    normal_points = []
    crash_points = []

    # 遍历去重后的数据
    for i, entry in enumerate(selection_log):
        obs_seq = obs_sequences[i]
        if not obs_seq: continue
        
        # obs_seq 是 [[x,y], [x,y]...] 列表
        seq_arr = np.array(obs_seq)
        
        # 简单的维度检查 (MountainCar通常是2维: position, velocity)
        if seq_arr.ndim != 2 or seq_arr.shape[1] < 2:
            continue

        is_crash = entry.get('crashed') or entry.get('did_crash', False)
        
        if is_crash:
            crash_points.append(seq_arr)
        else:
            normal_points.append(seq_arr)

    plt.figure(figsize=(12, 8))
    
    # 绘制正常轨迹 (灰色)
    if normal_points:
        all_normal = np.vstack(normal_points)
        # 如果点太多，随机采样以加快绘图
        if len(all_normal) > 100000:
            indices = np.random.choice(len(all_normal), 100000, replace=False)
            all_normal = all_normal[indices]
            
        plt.scatter(all_normal[:, 0], all_normal[:, 1], c='#B0BEC5', s=10, alpha=0.3, 
                    label='Normal Episodes', edgecolors='none', rasterized=True)

    # 绘制崩溃轨迹 (红色)
    if crash_points:
        all_crash = np.vstack(crash_points)
        plt.scatter(all_crash[:, 0], all_crash[:, 1], c='#D32F2F', s=20, alpha=0.8, 
                    label='Crash Episodes', marker='x')

    plt.title('State Space Coverage: Normal vs. Crash Episodes', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Position', fontsize=14, labelpad=10)
    plt.ylabel('Velocity', fontsize=14, labelpad=10)
    
    # MountainCar 的典型边界线
    plt.axvline(x=-1.2, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0.6, color='k', linestyle='--', alpha=0.3)
    
    plt.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=12)
    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    try:
        plt.savefig(PLOT_2_NAME, dpi=300)
        print(f"已保存: {PLOT_2_NAME}")
    except Exception as e:
        print(f"保存图表 2 失败: {e}")
    plt.close()

def plot_3_mutation_depth(selection_log):
    """[图表3] 变异代数直方图"""
    print(f"正在绘制: {PLOT_3_NAME}")
    
    crash_depths = []
    for entry in selection_log:
        is_crash = entry.get('crashed') or entry.get('did_crash', False)
        if is_crash:
            gen = entry.get('generation')
            if gen is None:
                p_depth = entry.get('parent_depth')
                if p_depth is not None:
                    gen = p_depth + 1
                else:
                    gen = 0 
            crash_depths.append(gen)
            
    if not crash_depths:
        print("[提示] 没有找到崩溃代数数据，跳过图表 3。")
        return

    mean_gen = np.mean(crash_depths)
    median_gen = np.median(crash_depths)
    max_gen = np.max(crash_depths)
    
    plt.figure(figsize=(12, 7))
    max_x = int(max_gen)
    bins = np.arange(0, max_x + 2) - 0.5 

    n, bins, patches = plt.hist(crash_depths, bins=bins, color='#009688', edgecolor='white', alpha=0.85, rwidth=0.8)
    
    plt.title('Distribution of Crashes by Mutation Generation', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Generation (Depth)', fontsize=14, labelpad=10)
    plt.ylabel('Count of Crashes', fontsize=14, labelpad=10)
    
    plt.xticks(range(0, max_x + 1))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 在柱子上显示数值
    for i in range(len(patches)):
        if n[i] > 0:
            plt.text(patches[i].get_x() + patches[i].get_width()/2, n[i], int(n[i]), 
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='#455A64')

    stats_text = (
        f"$\\bf{{Statistics}}$\n"
        f"Mean: {mean_gen:.2f}\n"
        f"Median: {median_gen:.1f}\n"
        f"Max: {int(max_gen)}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=13, 
                   verticalalignment='top', horizontalalignment='right', bbox=props)

    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    try:
        plt.savefig(PLOT_3_NAME, dpi=300)
        print(f"已保存: {PLOT_3_NAME}")
    except Exception as e:
        print(f"保存图表 3 失败: {e}")
    plt.close()

def plot_4_crashes_vs_inputs(selection_log):
    """[图表4] 累积崩溃数 vs 累积独特输入数"""
    print(f"正在绘制: {PLOT_4_NAME}")
    
    cumulative_crashes = []
    current_count = 0
    
    for entry in selection_log:
        is_crash = entry.get('crashed') or entry.get('did_crash', False)
        if is_crash:
            current_count += 1
        cumulative_crashes.append(current_count)
            
    if not cumulative_crashes:
        return

    iterations = range(1, len(cumulative_crashes) + 1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, cumulative_crashes, label='Cumulative Unique Crashes', color='#3F51B5', linewidth=2)
    plt.fill_between(iterations, cumulative_crashes, color='#3F51B5', alpha=0.1)
    
    plt.title('Unique Crashes Found vs. Unique Inputs Discovered', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Number of Unique Inputs Discovered', fontsize=14, labelpad=10)
    plt.ylabel('Cumulative Unique Crashes', fontsize=14, labelpad=10)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    
    sns.despine()
    plt.tight_layout()
    
    try:
        plt.savefig(PLOT_4_NAME, dpi=300)
        print(f"已保存: {PLOT_4_NAME}")
    except Exception as e:
        print(f"保存图表 4 失败: {e}")
    plt.close()

# --- 4. 主函数 ---

def main():
    # 1. 加载文件
    raw_log = load_pickle(LOG_FILE)
    raw_obs = load_obs_txt(OBS_FILE)
    
    if raw_log is None or raw_obs is None:
        print("必要的文件加载失败，程序终止。")
        return

    # 2. 去重并同步数据
    dedup_log, dedup_obs = deduplicate_data(raw_log, raw_obs)
    
    if not dedup_log:
        print("数据去重后为空，无法绘图。")
        return

    # 3. 绘制四张图表
    plot_1_crashes_over_time(dedup_log, len(raw_log))  # 图1：时间趋势
    plot_2_state_space(dedup_log, dedup_obs)           # 图2：状态空间 (使用 obs)
    plot_3_mutation_depth(dedup_log)                   # 图3：代数分布
    plot_4_crashes_vs_inputs(dedup_log)                # 图4：崩溃效率
        
    print("\n所有分析和绘图已完成。")

if __name__ == "__main__":
    main()