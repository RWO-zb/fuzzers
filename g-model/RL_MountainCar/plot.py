import pickle
import matplotlib.pyplot as plt

# 全局绘图风格配置
plt.style.use('ggplot')
plt.rcParams.update({'font.sans-serif': ['Arial'], 'axes.unicode_minus': False})

# 定义颜色：Random使用灰色，Generative使用橙色
COLORS = {'rand': '#95a5a6', 'gen': '#e67e22'}

def plot_bar_chart(rand_count, gen_count, total_samples, title_text, filename):
    """
    辅助函数：绘制并保存柱状图
    参数:
      rand_count: Random 方法发现的 Crash 数
      gen_count: Generative 方法发现的 Crash 数
      total_samples: 当前统计阶段的总测试样例数 (用于显示在标题中)
      title_text: 图表主标题
      filename: 保存的文件名
    """
    plt.figure(figsize=(8, 6), tight_layout=True)
    
    labels = ['Random', 'Generative']
    counts = [rand_count, gen_count]
    
    # 绘制柱状图
    bars = plt.bar(labels, counts, color=[COLORS['rand'], COLORS['gen']], width=0.5)
    
    # 在柱子上方标注具体数值
    plt.bar_label(bars, padding=3, fontsize=12, fontweight='bold')
    
    # 设置标题（包含总样本数信息）
    full_title = f"{title_text}\n(Total Test Cases: {total_samples})"
    plt.title(full_title, pad=15, fontsize=14)
    plt.ylabel('Crash Count')
    plt.grid(axis='x') # 仅显示横向网格
    
    # 保存文件
    plt.savefig(filename, dpi=300)
    print(f"图表已保存: {filename}")
    plt.close()

def main():
    # 1. 加载数据
    pkl_filename = "results\\MC_generative+novelty_50_seed_0\\all_test_cases_log.pkl"
    try:
        with open(pkl_filename, 'rb') as f:
            raw_data = pickle.load(f)
    except FileNotFoundError:
        print(f"错误：未找到文件 '{pkl_filename}'，请确保文件在当前目录下。")
        return

    # ==========================================
    # 统计阶段 1: 去重前 (Raw Data)
    # ==========================================
    total_raw = len(raw_data)  # 去重前的总测试样例数
    
    # 筛选出所有标记为 crash 的条目
    raw_crashes = [e for e in raw_data if e.get('is_crash')]
    
    # 分别统计 Random 和 Generative 的数量
    raw_crash_rand = sum(1 for c in raw_crashes if c.get('source') == 'random')
    raw_crash_gen = sum(1 for c in raw_crashes if c.get('source') == 'generative')

    print("=" * 60)
    print(f"【去重前 (Raw Data)】统计报告")
    print("-" * 60)
    print(f"  > 总测试样例数 (Total Samples) : {total_raw}")
    print("-" * 30)
    print(f"  - Random 触发 Crash 次数       : {raw_crash_rand}")
    print(f"  - Generative 触发 Crash 次数   : {raw_crash_gen}")
    print(f"  - 总 Crash 触发次数            : {len(raw_crashes)}")
    print("=" * 60)

    # 绘图 1: 去重前的 Crash 分布
    plot_bar_chart(
        raw_crash_rand, 
        raw_crash_gen, 
        total_raw,
        'Total Crashes Detected (Before Deduplication)', 
        'crash_counts_raw_before_dedup.png'
    )

    # ==========================================
    # 数据处理: 去重 (Deduplication)
    # ==========================================
    print("\n... 正在执行去重处理 ...\n")
    unique_data = []
    seen_inputs = set()
    
    for entry in raw_data:
        # 将 input 列表转换为元组，以便用于集合(Set)去重
        t_in = tuple(entry['input'])
        if t_in not in seen_inputs:
            seen_inputs.add(t_in)
            unique_data.append(entry)

    # ==========================================
    # 统计阶段 2: 去重后 (Unique Data)
    # ==========================================
    total_unique = len(unique_data)  # 去重后的总测试样例数

    # 筛选出去重后仍标记为 crash 的条目
    unique_crashes = [e for e in unique_data if e.get('is_crash')]
    
    # 分别统计 Random 和 Generative 的数量
    unique_crash_rand = sum(1 for c in unique_crashes if c.get('source') == 'random')
    unique_crash_gen = sum(1 for c in unique_crashes if c.get('source') == 'generative')

    print("=" * 60)
    print(f"【去重后 (Unique Data)】统计报告")
    print("-" * 60)
    print(f"  > 总独特测试样例数 (Total Unique) : {total_unique}")
    print("-" * 30)
    print(f"  - Random 发现的唯一 Crash         : {unique_crash_rand}")
    print(f"  - Generative 发现的唯一 Crash     : {unique_crash_gen}")
    print(f"  - 总唯一 Crash 数                 : {len(unique_crashes)}")
    print("=" * 60)

    # 绘图 2: 去重后的 Crash 分布
    plot_bar_chart(
        unique_crash_rand, 
        unique_crash_gen, 
        total_unique,
        'Unique Crashes Detected (After Deduplication)', 
        'crash_counts_unique_after_dedup.png'
    )

if __name__ == "__main__":
    main()