import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ==========================================
# 1. 定义计算 Cliff's Delta 的函数
# ==========================================
def cliffs_delta(x, y):
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.nan
    count_x_greater_y = sum(xi > yi for xi in x for yi in y)
    count_y_greater_x = sum(xi < yi for xi in x for yi in y)
    return (count_x_greater_y - count_y_greater_x) / (n * m)

# ==========================================
# 2. 读取数据 (解决编码与文件格式问题)
# ==========================================
try:
    # 尝试读取 CSV (适配你上传的 CSV 文件名)
    df = pd.read_csv('fuzzing_90_results.xlsx - Sheet1.csv', encoding='gbk')
except UnicodeDecodeError:
    df = pd.read_csv('fuzzing_90_results.xlsx - Sheet1.csv', encoding='utf-8')
except FileNotFoundError:
    df = pd.read_excel('fuzzing_90_results.xlsx')

# ==========================================
# 3. 提取数据与计算 Z-score
# 计算目标指标：Sec_Per_Crash
# ==========================================
target_col = 'Sec_Per_Crash'
tools = df['Tool'].unique()
data_dict = {tool: df[df['Tool'] == tool][target_col].values for tool in tools}

# 计算全局均值和标准差用于 Z-score
all_values = df[target_col].values
global_mean = np.mean(all_values)
global_std = np.std(all_values)

# 每个工具的 Z-score 均值
method_z_scores_mean = {tool: (np.mean(vals) - global_mean) / global_std for tool, vals in data_dict.items()}

# ==========================================
# 4. 类内分组 (完全保留原有 Wilcoxon 逻辑)
# 注意：因为时间越少越好，我们按 Z-score 从小到大排序
# ==========================================
sorted_methods = sorted(tools, key=lambda x: method_z_scores_mean[x])

groups = []
if sorted_methods:
    current_group = [sorted_methods[0]]
    for i in range(1, len(sorted_methods)):
        m1 = current_group[-1]
        m2 = sorted_methods[i]
        
        # 执行 Wilcoxon 符号秩检验
        try:
            stat, p_val = wilcoxon(data_dict[m1], data_dict[m2])
        except ValueError: # 如果数据完全一致
            p_val = 1.0
            
        if p_val > 0.05:
            current_group.append(m2)
        else:
            groups.append(current_group)
            current_group = [m2]
    groups.append(current_group)

# 转换分组结果
group_list = []
for idx, g in enumerate(groups):
    # 生成字母标签 'a', 'b', 'c', 'd' 等，idx=0 即为 'a'
    group_letter = chr(ord('a') + idx) 
    for m in g:
        group_list.append({
            'Group': group_letter,
            'Tool': m,
            f'Mean_{target_col}': round(np.mean(data_dict[m]), 2),
            'Z-Score': round(method_z_scores_mean[m], 4)
        })
group_df = pd.DataFrame(group_list)

# ==========================================
# 5. 类间比较 (完全保留原有 Cliff's Delta 逻辑)
# ==========================================
inter_class_results = []
for i in range(len(groups) - 1):
    g1_letter = chr(ord('a') + i)
    g2_letter = chr(ord('a') + i + 1)
    m1, m2 = groups[i][0], groups[i+1][0]
    
    delta = cliffs_delta(data_dict[m1], data_dict[m2])
    
    abs_d = abs(delta)
    if abs_d >= 0.474: magnitude = 'Large'
    elif abs_d >= 0.33: magnitude = 'Medium'
    elif abs_d >= 0.147: magnitude = 'Small'
    else: magnitude = 'Negligible'
    
    inter_class_results.append({
        'Group Comparison': f"Group {g1_letter} vs Group {g2_letter}",
        'Represented By': f"{m1} vs {m2}",
        'Cliff\'s Delta': round(delta, 4),
        'Effect Magnitude': magnitude
    })

inter_class_df = pd.DataFrame(inter_class_results)

# ==========================================
# 6. 打印结果并保存到文件
# ==========================================
# 构建准备输出的字符串文本
output_text = f"=== 基于 {target_col} 的类内分组 (时间越少越优，'a'组表现最好) ===\n"
output_text += group_df.to_string(index=False) + "\n\n"

output_text += "=== 基于 Effect Size 的类间比较 ===\n"
if not inter_class_df.empty:
    output_text += inter_class_df.to_string(index=False) + "\n"
else:
    output_text += "所有方法均被分入同一组。\n"

# 1. 打印到控制台
print(output_text)

# 2. 保存为纯文本报告 (.txt)
with open('statistical_results_report.txt', 'w', encoding='utf-8') as f:
    f.write(output_text)

# 3. 保存为结构化的 CSV 表格，方便 Excel 直接打开 (.csv)
# 使用 utf-8-sig 编码，防止 Windows 下 Excel 打开含中文或特殊字符乱码
group_df.to_csv('grouped_results.csv', index=False, encoding='utf-8-sig')
if not inter_class_df.empty:
    inter_class_df.to_csv('inter_class_results.csv', index=False, encoding='utf-8-sig')

print("\n[✓] 结果已成功保存到当前目录下的文件中：")
print("    - statistical_results_report.txt (排版好的纯文本报告)")
print("    - grouped_results.csv (包含 Tool、Group 等信息的分组表格)")
if not inter_class_df.empty:
    print("    - inter_class_results.csv (类间比较及效应量表格)")