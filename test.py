import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ==========================================
# 1. 定义计算 Cliff's Delta 的函数
# (注意: Cliff's Delta 评估的是整体分布的重叠度，依然适用)
# ==========================================
def cliffs_delta(x, y):
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.nan
    count_x_greater_y = sum(xi > yi for xi in x for yi in y)
    count_y_greater_x = sum(xi < yi for xi in x for yi in y)
    return (count_x_greater_y - count_y_greater_x) / (n * m)

# ==========================================
# 2. 鲁棒地读取并解析原始数据
# ==========================================
data_dict = {}
current_method = None
current_values = []

with open('p-value.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    # 清理掉多余的标记
    text = text.replace("", "").replace("", "")
    tokens = text.split()
    
    for token in tokens:
        try:
            val = float(token)
            if current_method is not None:
                current_values.append(val)
        except ValueError:
            if current_method is not None and len(current_values) > 0:
                data_dict[current_method] = np.array(current_values)
            current_method = token
            current_values = []
            
    if current_method is not None and len(current_values) > 0:
        data_dict[current_method] = np.array(current_values)

methods = list(data_dict.keys())

# ==========================================
# 3. 核心：局部标准化 (Z-score)
# ==========================================
records = []
for method, values in data_dict.items():
    if len(values) != 15:
        print(f"⚠️ 警告: 方法 '{method}' 数据量不为 15，请检查！")
        continue
    for i in range(3): # 3个数据集
        for j in range(5): # 每个数据集 5 个相同的随机种子
            records.append({
                'Method': method,
                'Dataset': f'Dataset_{i+1}',
                'Seed_Run': f'Seed_{j+1}', # 明确标识这是相同的种子
                'Raw_Score': values[i*5 + j]
            })

df_raw = pd.DataFrame(records)

# 按数据集分组计算 Z-score (抹平数据集之间的绝对量级差异)
df_raw['Z_Score'] = df_raw.groupby('Dataset')['Raw_Score'].transform(lambda x: (x - x.mean()) / x.std())

# 提取标准化后的数据供后续检验使用 (严格保持 15 次运行的顺序对齐)
z_data_dict = {m: df_raw[df_raw['Method'] == m]['Z_Score'].values for m in methods}
method_z_scores_mean = {m: np.mean(z_data_dict[m]) for m in methods}

# 按标准化均值得分排序（从高到低）
sorted_methods = sorted(methods, key=lambda x: method_z_scores_mean[x], reverse=True)

# ==========================================
# 4. 类内分组 (使用 Wilcoxon Signed-Rank 检验处理配对数据)
# ==========================================
group_labels = {method: set() for method in methods}
group_index = -1 

for i, method1 in enumerate(sorted_methods):
    if group_labels[method1]:  
        current_group = max(group_labels[method1]) 
    else:
        group_index += 1
        current_group = chr(ord('a') + group_index) 
        group_labels[method1].add(current_group)

    for method2 in sorted_methods[i + 1:]:
        can_join_group = True  
        for method_in_group in [m for m in methods if current_group in group_labels[m]]:
            if method2 == method_in_group:
                continue
            
            data_A = z_data_dict[method2]
            data_B = z_data_dict[method_in_group]
            
            # 使用配对样本的 Wilcoxon 符号秩检验
            try:
                # 只有当两次运行存在差异时才能计算 (避免完全相同的数据导致报错)
                if np.all(data_A == data_B):
                    p_val = 1.0
                else:
                    _, p_val = wilcoxon(data_A, data_B, alternative='two-sided')
            except Exception as e:
                p_val = 1.0 # 如果遇到极端异常无法计算，默认无显著差异
            
            # P-value < 0.05 认为有显著差异，不能同组
            if p_val < 0.05:
                can_join_group = False  
                break  

        if can_join_group:
            group_labels[method2].add(current_group)

final_groups = {method: "".join(sorted(group_labels[method])) for method in methods}

# 生成基础分组表
group_df = pd.DataFrame.from_dict(final_groups, orient='index', columns=['Class (Group)'])
group_df['Mean Z-Score'] = group_df.index.map(method_z_scores_mean).round(4)
group_df.index.name = 'Approach'
group_df = group_df.reindex(sorted_methods)

# ==========================================
# 5. 类间比较 (基于配对检验的 P-value 与 Effect Size)
# ==========================================
class_representatives = {}
for method, group in final_groups.items():
    primary_group = group[0] 
    if primary_group not in class_representatives:
        class_representatives[primary_group] = method

inter_class_results = []
groups_list = sorted(list(class_representatives.keys()))

for i in range(len(groups_list)):
    for j in range(i + 1, len(groups_list)):
        g1, g2 = groups_list[i], groups_list[j]
        m1, m2 = class_representatives[g1], class_representatives[g2]
        
        data1, data2 = z_data_dict[m1], z_data_dict[m2]
        
        try:
            if np.all(data1 == data2):
                p_val = 1.0
            else:
                _, p_val = wilcoxon(data1, data2, alternative='two-sided')
        except:
            p_val = 1.0
            
        delta = cliffs_delta(data1, data2)
        
        abs_d = abs(delta)
        if abs_d >= 0.474: magnitude = 'Large'
        elif abs_d >= 0.33: magnitude = 'Medium'
        elif abs_d >= 0.147: magnitude = 'Small'
        else: magnitude = 'Negligible'
        
        inter_class_results.append({
            'Class Comparison': f"Class '{g1}' vs Class '{g2}'",
            'Represented By': f"{m1} vs {m2}",
            'Z-Score Diff': round(method_z_scores_mean[m1] - method_z_scores_mean[m2], 4),
            'p_value': p_val,
            "Cliff's Delta": round(delta, 4),
            'Effect Magnitude': magnitude
        })

inter_class_df = pd.DataFrame(inter_class_results)

# ==========================================
# 6. 打印与保存
# ==========================================
print("\n=== 基于 Z-score 与 配对 Wilcoxon 检验 的类内分组 ===")
print(group_df)
print("\n=== 基于 Effect Size 的类间比较 ===")
if len(inter_class_df) > 0:
    print(inter_class_df[['Class Comparison', 'Represented By', "Cliff's Delta", 'Effect Magnitude']].to_string(index=False))
else:
    print("所有方法均被分入了同一个组，无类间比较结果。")

output_file = './paired_wilcoxon_analysis.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    group_df.to_excel(writer, sheet_name='Intra-Class (Groups)')
    if len(inter_class_df) > 0:
        inter_class_df.to_excel(writer, sheet_name='Inter-Class (Effect Size)', index=False)
    df_raw.to_excel(writer, sheet_name='Raw & Z-score Data', index=False)

print(f"\n✅ 结果已保存至: {output_file}")