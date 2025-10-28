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
# '1761565216.2574737_data.csv' 是您上传的数据文件名
input_filename = "1761565216.2574737_data.csv"
# 修改输出文件名以反映去重
plot_filename = 'fuzz_crashes_vs_total_cases_dedup.png'
# 新的去重后数据文件名
dedup_csv_filename = 'data_deduplicated.csv'

try:
    # --- 数据加载 ---
    # 打开并读取 CSV 数据文件
    df = pd.read_csv(input_filename)
    print(f"原始数据行数: {len(df)}")

    # --- 新增：数据去重 ---
    # 假设基于所有列去除完全重复的行
    # 使用一个新的变量 df_dedup 来存储去重后的数据
    df_dedup = df.drop_duplicates()
    print(f"去重后数据行数: {len(df_dedup)}")
    print(f"移除了 {len(df) - len(df_dedup)} 行重复数据。")
    
    # 将去重后的数据保存到新文件
    df_dedup.to_csv(dedup_csv_filename, index=False)
    print(f"去重后的数据已保存到: {dedup_csv_filename}")


    # 检查是否收集到了数据
    if df_dedup.empty:
        print(f"未从文件 '{input_filename}' 处理任何数据（或去重后为空）。无法生成绘图。")
    else:
        # --- 数据处理 (使用去重后的 df_dedup) ---
        
        # 1. 创建 'TotalCasesTested' 列
        # 值为 1, 2, 3, ..., N (N=去重后的总行数)
        # 这代表 *唯一* 测试用例的数量
        # 需要使用 .copy() 来避免 SettingWithCopyWarning
        df_processed = df_dedup.copy()
        df_processed['TotalCasesTested'] = range(1, len(df_processed) + 1)
        
        # 2. 创建 'CrashesFound' 列
        # 'is_faulty' 是 bool 类型, (True, False)
        # 将 bool 转换为 int (True=1, False=0)，然后计算累计和
        # 这代表在唯一测试用例中发现的累计崩溃数
        df_processed['CrashesFound'] = df_processed['is_faulty'].astype(int).cumsum()

        print("\n去重后数据处理完成。DataFrame 信息:")
        df_processed.info()
        print("\nDataFrame 头部:")
        print(df_processed.head())
        print("\nDataFrame 尾部 (显示最终计数):")
        print(df_processed.tail())

        # --- 绘图 (仿照示例代码样式) ---
        # 直接绘图，不使用 plt.figure() 以兼容虚拟机环境
        plt.plot(df_processed['TotalCasesTested'], df_processed['CrashesFound'], marker='.', linestyle='-', markersize=2, label='Cumulative Crashes (Deduplicated)')

        # 设置图表标题和坐标轴标签 (更新为 "Unique")
        plt.title('Cumulative Crashes Found vs. Unique Cases Tested')
        plt.xlabel('Total Unique Cases Tested')
        plt.ylabel('Cumulative Crashes Found')

        # 添加网格
        plt.grid(True)

        # 添加图例
        plt.legend()

        # 确保布局良好
        plt.tight_layout()

        # 保存图表到文件
        plt.savefig(plot_filename)
        
        print(f"\n绘图已保存到: {plot_filename}")

except FileNotFoundError:
    print(f"错误: 输入文件 '{input_filename}' 未找到。")
except Exception as e:
    print(f"处理数据或绘图时发生错误: {e}")