import pandas as pd
import matplotlib.pyplot as plt
import json  # 用于处理 JSON 文件

# 尝试设置一个通用的默认字体来处理减号
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"无法设置 DejaVu Sans 字体: {e}。使用默认字体。")

# 用于存储绘图数据点的列表
data = []
# 初始化计数器
total_cases_tested = 0
crashes_found = 0

# 定义输入和输出文件名
input_filename = 'D:\\code\\fuzzers\\g-model\\RL_BipedalWalker\\results\\all_fuzz_cases_log.json'
plot_filename = 'D:\\code\\fuzzers\\g-model\\RL_BipedalWalker\\results\\fuzz_crashes_vs_total_cases.png'

try:
    # 打开并读取 JSON 数据文件
    with open(input_filename, 'r', encoding='utf-8') as f:
        json_data = json.load(f)  # 加载整个 JSON 数据

    # 检查是否收集到了数据
    if not json_data:
        print(f"未从文件 '{input_filename}' 处理任何数据。无法生成绘图。")
    else:
        # 遍历 JSON 数据中的每个测试用例
        for test_case in json_data:
            # 每次循环（即每个测试用例）增加测试总数
            total_cases_tested += 1

            # 检查 "status" 字段是否为 "failure"，将其视为一次崩溃
            if test_case.get('status') == 'failure':
                crashes_found += 1

            # 将当前的累计计数附加到数据列表中用于绘图
            data.append({'TotalCasesTested': total_cases_tested, 'CrashesFound': crashes_found})

        # --- 数据处理与绘图 ---
        
        # 从收集的数据创建 pandas DataFrame
        df = pd.DataFrame(data)

        print("数据处理完成。DataFrame 信息:")
        df.info()
        print("\nDataFrame 头部:")
        print(df.head())
        print("\nDataFrame 尾部 (显示最终计数):")
        print(df.tail())

        # --- 绘图 ---
        # 直接绘图，不使用 plt.figure() 以兼容虚拟机环境
        plt.plot(df['TotalCasesTested'], df['CrashesFound'], marker='.', linestyle='-', markersize=2, label='Cumulative Crashes')

        # 设置图表标题和坐标轴标签
        plt.title('Cumulative Crashes Found vs. Total Fuzz Cases Tested')
        plt.xlabel('Total Fuzz Cases Tested')
        plt.ylabel('Cumulative Crashes Found')

        # 添加网格以便更好地阅读
        plt.grid(True)
        # 添加图例
        plt.legend()
        # 调整布局以防止标签/标题重叠
        plt.tight_layout()

        # 将图表保存到文件
        plt.savefig(plot_filename)
        print(f"\n绘图已成功保存为: {plot_filename}")

# 处理潜在的错误
except FileNotFoundError:
    print(f"错误: 未找到文件 '{input_filename}'。")
except json.JSONDecodeError:
    print(f"错误: 无法解码 '{input_filename}' 中的 JSON 数据。")
except Exception as e:
    print(f"在处理或绘图过程中发生错误: {e}")