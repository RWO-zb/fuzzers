import pickle
import matplotlib.pyplot as plt
import os

# 全局绘图风格配置
plt.style.use('ggplot')
plt.rcParams.update({'font.sans-serif': ['Arial'], 'axes.unicode_minus': False})

# 定义颜色：Random使用灰色，Generative使用橙色
COLORS = {'rand': '#95a5a6', 'gen': '#e67e22'}

def plot_bar_chart(rand_count, gen_count, total_samples, title_text, filename):
    """
    辅助函数：绘制并保存柱状图
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

def plot_cumulative_curve(unique_crashes, filename):
    """
    新增函数：绘制 Unique Crash 随时间变化的累积曲线
    """
    # 1. 安全检查：是否有数据
    if not unique_crashes:
        print("警告：没有 Unique Crash 数据，跳过绘制时间曲线。")
        return

    # 2. 安全检查：是否有 'time' 字段
    # 我们检查第一条数据（假设所有数据结构一致）
    if 'time' not in unique_crashes[0]:
        print("警告：日志数据中缺少 'time' 字段，无法绘制随时间变化的曲线。")
        print("请确保您运行的是更新后的 test_gen.py (包含时间戳记录)。")
        return

    # 3. 数据准备
    # 按时间排序 (以防日志顺序并不是严格的时间序)
    sorted_crashes = sorted(unique_crashes, key=lambda x: x['time'])
    
    # 提取 X 轴 (Time in Hours) 和 Y 轴 (Cumulative Count)
    # 将秒转换为小时
    times = [item['time'] / 3600.0 for item in sorted_crashes]
    counts = list(range(1, len(sorted_crashes) + 1))
    
    # 4. 绘图
    plt.figure(figsize=(10, 6), tight_layout=True)
    
    # 绘制主曲线 (深蓝色连线)
    plt.plot(times, counts, color='#2c3e50', linewidth=2, label='Total Unique Crashes', zorder=2)
    
    # 分离 Random 和 Generative 的点，以便用不同颜色标记
    rand_x, rand_y = [], []
    gen_x, gen_y = [], []
    
    for i, item in enumerate(sorted_crashes):
        t = times[i]
        c = counts[i]
        if item.get('source') == 'random':
            rand_x.append(t)
            rand_y.append(c)
        else:
            gen_x.append(t)
            gen_y.append(c)
            
    # 绘制散点 (在曲线之上 zorder=3)
    # 这让我们可以清楚地看到每个 Crash 是由谁发现的
    if rand_x:
        plt.scatter(rand_x, rand_y, color=COLORS['rand'], s=50, label='Random', zorder=3, edgecolors='white')
    if gen_x:
        plt.scatter(gen_x, gen_y, color=COLORS['gen'], s=50, label='Generative', zorder=3, edgecolors='white')
        
    # 5. 图表修饰
    plt.title('Cumulative Unique Crashes Over Time', fontsize=14, pad=15)
    plt.xlabel('Time (Hours)', fontsize=12)
    plt.ylabel('Unique Crashes Found', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存
    plt.savefig(filename, dpi=300)
    print(f"图表已保存: {filename}")
    plt.close()

def main():
    # 1. 加载数据
    # 注意：请确保此路径与您 test_gen.py 生成结果的文件夹一致
    # 建议使用 os.path.join 兼容不同操作系统
    pkl_filename = os.path.join("results", "generative+novelty_50_seed_0", "all_test_cases_log.pkl")
    
    if not os.path.exists(pkl_filename):
        print(f"错误：未找到文件 '{pkl_filename}'。")
        print("提示：请检查 `pkl_filename` 路径是否正确，或确保您已运行过 test_gen.py。")
        return

    try:
        with open(pkl_filename, 'rb') as f:
            raw_data = pickle.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # ==========================================
    # 统计阶段 1: 去重前 (Raw Data)
    # ==========================================
    total_raw = len(raw_data)
    raw_crashes = [e for e in raw_data if e.get('is_crash')]
    
    raw_crash_rand = sum(1 for c in raw_crashes if c.get('source') == 'random')
    raw_crash_gen = sum(1 for c in raw_crashes if c.get('source') == 'generative')

    print("=" * 60)
    print(f"【去重前 (Raw Data)】统计报告")
    print("-" * 60)
    print(f"  > 总测试样例数 : {total_raw}")
    print(f"  - Random Crash : {raw_crash_rand}")
    print(f"  - Gen Crash    : {raw_crash_gen}")
    print("=" * 60)

    plot_bar_chart(
        raw_crash_rand, raw_crash_gen, total_raw,
        'Total Crashes Detected (Before Deduplication)', 
        'crash_counts_raw.png'
    )

    # ==========================================
    # 数据处理: 去重 (Deduplication)
    # ==========================================
    print("\n... 正在执行去重处理 ...\n")
    unique_data = []
    seen_inputs = set()
    
    # 保持时间顺序进行去重
    for entry in raw_data:
        t_in = tuple(entry['input'])
        if t_in not in seen_inputs:
            seen_inputs.add(t_in)
            unique_data.append(entry)

    # ==========================================
    # 统计阶段 2: 去重后 (Unique Data)
    # ==========================================
    total_unique = len(unique_data)
    unique_crashes = [e for e in unique_data if e.get('is_crash')]
    
    unique_crash_rand = sum(1 for c in unique_crashes if c.get('source') == 'random')
    unique_crash_gen = sum(1 for c in unique_crashes if c.get('source') == 'generative')

    print("=" * 60)
    print(f"【去重后 (Unique Data)】统计报告")
    print("-" * 60)
    print(f"  > 总独特样例数 : {total_unique}")
    print(f"  - Random Crash : {unique_crash_rand}")
    print(f"  - Gen Crash    : {unique_crash_gen}")
    print(f"  - Total Unique : {len(unique_crashes)}")
    print("=" * 60)

    plot_bar_chart(
        unique_crash_rand, unique_crash_gen, total_unique,
        'Unique Crashes Detected (After Deduplication)', 
        'crash_counts_unique.png'
    )
    
    # ==========================================
    # 绘图 3: Unique Crash 随时间变化的曲线
    # ==========================================
    plot_cumulative_curve(unique_crashes, 'unique_crashes_over_time.png')

if __name__ == "__main__":
    main()