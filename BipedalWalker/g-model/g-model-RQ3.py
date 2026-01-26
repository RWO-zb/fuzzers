import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

# ================= 配置参数 =================
LOG_FILE = "all_test_cases_log.pkl"  # 日志文件路径
STEP_SIZE = 50                       # 对应 args.step，默认 50
PLOT_FILENAME = "crash_generation_analysis.png" # 保存的图片名称
# ===========================================

def load_data(file_path):
    """加载 Pickle 日志数据"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"[Info] Successfully loaded {len(data)} entries from {file_path}")
            return data
    except FileNotFoundError:
        print(f"[Error] File {file_path} not found.")
        return []

def calculate_generations(data, step_size):
    """
    提取唯一 Crash 并计算其代数
    定义: 0-50步=第1代, 51-100步=第2代...
    """
    unique_crashes = set()
    crash_generations = []
    
    count_total_crashes = 0
    
    for entry in data:
        # 1. 检查是否是 Crash
        if entry.get('is_crash', False):
            count_total_crashes += 1
            
            # 2. 获取 Input 并转换为 Tuple 以便哈希去重
            inp = entry.get('input')
            if inp is None:
                continue
            
            # 兼容 list 和 numpy array
            if isinstance(inp, list):
                inp_key = tuple(inp)
            elif isinstance(inp, np.ndarray):
                inp_key = tuple(inp.tolist())
            else:
                inp_key = inp # 假设是 bytes 或其他 hashable
                
            # 3. 如果是新的 Unique Crash，计算代数
            if inp_key not in unique_crashes:
                unique_crashes.add(inp_key)
                
                step = entry.get('step', 0)
                
                # --- 代数计算核心逻辑 ---
                # step 0-50 -> ceil(<=1) -> 1
                # step 51-100 -> ceil(1.02...2.0) -> 2
                generation = max(1, math.ceil(step / step_size))
                
                crash_generations.append(generation)

    print(f"[Info] Total Crashes: {count_total_crashes}")
    print(f"[Info] Unique Crashes: {len(unique_crashes)}")
    return crash_generations

def calculate_five_number_summary(data):
    """计算五数概括 + 平均值"""
    if not data:
        return None
    
    # 五数概括: Min, Q1, Median, Q3, Max
    minimum = np.min(data)
    q1 = np.percentile(data, 25)
    median = np.median(data)
    q3 = np.percentile(data, 75)
    maximum = np.max(data)
    mean_val = np.mean(data)
    
    return {
        "Min": minimum,
        "Q1": q1,
        "Median": median,
        "Q3": q3,
        "Max": maximum,
        "Mean": mean_val
    }

def plot_results(generations, stats):
    """绘制箱线图和小提琴图"""
    if not generations:
        print("[Warn] No generation data to plot.")
        return

    # 设置风格
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. 绘制箱线图 (Boxplot)
    # patch_artist=True 允许填充颜色
    box = ax.boxplot(generations, vert=False, patch_artist=True, 
                     showmeans=True,  # 显示平均值点
                     widths=0.6,
                     meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":10})

    # 美化箱线图颜色
    colors = ['#3498db']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # 2. 叠加散点图 (Jitter Plot) 以显示具体分布密度
    y = np.random.normal(1, 0.04, size=len(generations))
    ax.scatter(generations, y, alpha=0.5, color='#e74c3c', s=10, label='Unique Crash Instance')

    # 设置标签和标题
    ax.set_title('Distribution of Unique Crashes by Generation', fontsize=16, fontweight='bold')
    ax.set_xlabel('Generation Number', fontsize=14)
    ax.set_yticks([]) # 隐藏 Y 轴刻度，因为只有一个类别
    
    # 标注五数概括文本
    stats_text = (
        f"Mean:   {stats['Mean']:.2f}\n"
        f"Max:    {stats['Max']:.0f}\n"
        f"Q3:     {stats['Q3']:.0f}\n"
        f"Median: {stats['Median']:.0f}\n"
        f"Q1:     {stats['Q1']:.0f}\n"
        f"Min:    {stats['Min']:.0f}"
    )
    
    # 将统计数据放在图表右上角
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # X轴刻度优化 (显示整数)
    plt.xticks(np.arange(min(generations), max(generations)+2, step=max(1, int(max(generations)/10))))

    plt.tight_layout()
    plt.savefig(PLOT_FILENAME, dpi=300)
    print(f"[Success] Plot saved to {PLOT_FILENAME}")
    # plt.show() # 如果在 Jupyter 环境可取消注释

def main():
    # 1. 加载数据
    data = load_data(LOG_FILE)
    if not data:
        return

    # 2. 计算代数
    generations = calculate_generations(data, STEP_SIZE)
    
    if not generations:
        print("[Warn] No unique crashes found.")
        return

    # 3. 统计指标
    stats = calculate_five_number_summary(generations)
    print("\n=== Statistical Summary of Crash Generations ===")
    for k, v in stats.items():
        print(f"{k:<10}: {v:.2f}")
        
    # 4. 绘图
    plot_results(generations, stats)

if __name__ == "__main__":
    main()