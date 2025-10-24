import pandas as pd
import matplotlib.pyplot as plt
import json

# 尝试设置一个通用的默认字体来处理减号
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"无法设置 DejaVu Sans 字体: {e}。使用默认字体。")

# 定义输入和输出文件名
# '1760365542.9026809_data.csv' 是您上传的数据文件名
input_filename = "1760365542.9026809_data.csv" 
plot_filename = 'fuzz_crashes_vs_total_cases.png'

try:
    # --- 数据加载 ---
    # 打开并读取 CSV 数据文件
    df = pd.read_csv(input_filename)

    # 检查是否收集到了数据
    if df.empty:
        print(f"未从文件 '{input_filename}' 处理任何数据。无法生成绘图。")
    else:
        # --- 数据处理 ---
        
        # 1. 创建 'TotalCasesTested' 列
        # 值为 1, 2, 3, ..., N (N=总行数)
        # 模拟示例代码中遍历每个测试用例并递增计数器
        df['TotalCasesTested'] = range(1, len(df) + 1)
        
        # 2. 创建 'CrashesFound' 列
        # 'is_faulty' 是 bool 类型, (True, False)
        # 将 bool 转换为 int (True=1, False=0)，然后计算累计和
        # 模拟示例代码中检查 'status' == 'failure' 并递增计数器
        df['CrashesFound'] = df['is_faulty'].astype(int).cumsum()

        print("数据处理完成。DataFrame 信息:")
        df.info()
        print("\nDataFrame 头部:")
        print(df.head())
        print("\nDataFrame 尾部 (显示最终计数):")
        print(df.tail())

        # --- 绘图 (仿照示例代码样式) ---
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
        
        # 调整布局以防止标签被截断
        plt.tight_layout()

        # 保存图表到文件
        plt.savefig(plot_filename)
        
        print(f"\n绘图已保存到 '{plot_filename}'")

except FileNotFoundError:
    print(f"错误: 文件 '{input_filename}' 未找到。")
except Exception as e:
    print(f"处理数据或绘图时发生错误: {e}")