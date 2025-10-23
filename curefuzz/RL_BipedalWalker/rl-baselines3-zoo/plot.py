import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 为了正确显示中文，尝试设置中文字体
# 注意：虚拟机环境中不一定有这些字体，这是一个尽力而为的尝试。
# 我们将添加一个备用方案，如果找不到特定字体，则使用默认字体。
try:
    # 尝试使用常见的黑体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
except Exception as e:
    print(f"Could not set SimHei font: {e}. Using default font.")

# 存储提取的数据
data = []
# 正则表达式匹配 "Total seeds tested: X, Crashes found: Y"
pattern = re.compile(r"Total seeds tested: (\d+), Crashes found: (\d+)")

try:
    # 打开并读取文本文件
    with open('D:\\code\\fuzzers\\curefuzz\\RL_BipedalWalker\\rl-baselines3-zoo\\results10_23_2025_18_46_59\\cure_fuzz.txt', 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # 提取数据并转换为整数
                seeds_tested = int(match.group(1))
                crashes_found = int(match.group(2))
                data.append({'TotalSeedsTested': seeds_tested, 'CrashesFound': crashes_found})

    if not data:
        print("未能从文件中提取到任何数据。")
    else:
        # 使用提取的数据创建 pandas DataFrame
        df = pd.DataFrame(data)
        
        print("数据提取成功。DataFrame 信息：")
        df.info()
        print("\n数据前五行：")
        print(df.head())

        # --- 开始绘图 ---
        plt.figure(figsize=(10, 6))
        # 绘制折线图
        plt.plot(df['TotalSeedsTested'], df['CrashesFound'], marker='.', linestyle='-', markersize=4)
        
        # 设置中文标签和标题
        plt.title('崩溃数量随测试种子总数的变化 (Crashes Found vs. Total Seeds Tested)')
        plt.xlabel('总测试种子数量 (Total Seeds Tested)')
        plt.ylabel('发现的崩溃数量 (Crashes Found)')
        
        plt.grid(True) # 添加网格
        plt.tight_layout() # 自动调整布局
        
        # 保存图表为图片文件
        plot_filename = 'crashes_vs_seeds.png'
        plt.savefig(plot_filename)
        print(f"图表已保存为: {plot_filename}")

except FileNotFoundError:
    print("错误：文件 'cure_fuzz.txt' 未找到。")
except Exception as e:
    print(f"处理文件或绘图时发生错误: {e}")