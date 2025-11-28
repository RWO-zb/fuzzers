import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 全局绘图风格配置
plt.style.use('ggplot')
plt.rcParams.update({'font.sans-serif': ['Arial'], 'axes.unicode_minus': False})
COLORS = {'safe': '#3498db', 'crash': '#e74c3c', 'rand': '#95a5a6', 'gen': '#e67e22', 'line': '#9b59b6'}

def main():
    # 1. 加载与去重 (Happy Path: 假设文件存在且格式正确)
    with open("results\\generative+novelty_50_seed_0\\all_test_cases_log.pkl", 'rb') as f:
        raw_data = pickle.load(f)

    data, seen = [], set()
    for entry in raw_data:
        t_in = tuple(entry['input'])
        if t_in not in seen:
            seen.add(t_in)
            data.append(entry)

    # 提取 Crash 集合
    crashes = [e for e in data if e['is_crash']]

    # --- 图表 1: 来源统计柱状图 (Random vs Generative) ---
    plt.figure(figsize=(8, 6), tight_layout=True)
    
    sources = ['random', 'generative']
    counts = [sum(1 for c in crashes if c['source'] == s) for s in sources]
    labels = ['Random', 'Generative']
    
    bars = plt.bar(labels, counts, color=[COLORS['rand'], COLORS['gen']], width=0.5)
    plt.bar_label(bars, padding=3, fontsize=12, fontweight='bold')
    
    plt.title('Unique Crashes by Method', pad=15)
    plt.ylabel('Count')
    plt.grid(axis='x')
    plt.savefig('crash_by_source.png', dpi=300)
    plt.close()

    # --- 图表 2: t-SNE 状态空间分布 ---
    # 假设 data 非空，直接处理
    inputs = np.array([e['input'] for e in data])
    labels = np.array([1 if e['is_crash'] else 0 for e in data])
    
    # 自动计算 perplexity
    tsne = TSNE(n_components=2, perplexity=min(30, len(inputs)-1), random_state=42, init='pca', learning_rate='auto')
    emb = tsne.fit_transform(inputs)

    plt.figure(figsize=(10, 8), tight_layout=True)
    # 绘制 Safe 点 (蓝色, 透明)
    plt.scatter(emb[labels==0, 0], emb[labels==0, 1], c=COLORS['safe'], alpha=0.2, s=20, label='Safe')
    # 绘制 Crash 点 (红色, 显眼)
    plt.scatter(emb[labels==1, 0], emb[labels==1, 1], c=COLORS['crash'], alpha=0.8, s=40, marker='x', label='Crash', linewidth=1.5)
    
    plt.title('State Space Visualization')
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    plt.savefig('test_gen_input_space_tsne.png', dpi=300)
    plt.close()

    # --- 图表 3: 随时间发现的 Crash 累计图 ---
    # 假设每条日志都有 time 字段
    times = sorted([c['time'] / 3600 for c in crashes])
    
    # 添加 (0,0) 起点使曲线从原点出发
    x = [0] + times
    y = [0] + list(range(1, len(times) + 1))

    plt.figure(figsize=(10, 6), tight_layout=True)
    plt.plot(x, y, color=COLORS['line'], linewidth=2.5)
    plt.fill_between(x, y, color=COLORS['line'], alpha=0.15)
    
    plt.title('Cumulative Crashes Detected Over Time')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Unique Crashes')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig('crash_over_time.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()