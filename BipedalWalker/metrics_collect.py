import re
import pandas as pd
import os

def parse_fuzzing_logs(input_file, output_excel):
    if not os.path.exists(input_file):
        print(f"Error: 找不到日志文件 {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 利用报告的头部作为分隔符，将多次运行的输出切分为独立的块
    # [1:] 是为了跳过第一个表头前面的无用部分
    blocks = re.split(r'Academic-Grade Crash & Diversity Analysis.*?\n={10,}', content)
    
    results = []
    
    for i, block in enumerate(blocks[1:], start=1):
        row = {'Run ID': f'Run {i}'}
        
        # 辅助提取函数，应用正则表达式抓取对应的数值
        def extract(pattern, text, is_int=False):
            match = re.search(pattern, text)
            if match:
                # 兼容带有正负号的小数（例如 Silhouette 可能为负）
                val = match.group(1)
                return int(val) if is_int else float(val)
            return None

        # 1. 基础性能与覆盖率指标
        row['Total Mutations'] = extract(r'Total Mutations Executed:\s+(\d+)', block, True)
        row['Valid Crashes'] = extract(r'Valid Crash Mutations:\s+(\d+)', block, True)
        row['Hit Ratio (%)'] = extract(r'Hit Ratio \(Valid Rate\):\s+([0-9.]+)%', block)
        row['Space Coverage (%)'] = extract(r'State Space Coverage:\s+([0-9.]+)%', block)
        row['Overhead Ratio (%)'] = extract(r'Fuzzer Overhead Ratio:\s+([0-9.]+)%', block)
        
        # 2. 崩溃效率与生存深度
        row['Unique Crashes'] = extract(r'Total Unique Crashes Discovered:\s+(\d+)', block, True)
        row['Mean Interval (hrs)'] = extract(r'Mean Interval per Crash:\s+([0-9.]+)', block)
        row['Mean Survival Steps'] = extract(r'Survival Steps \(Depth\) - Mean:\s+([0-9.]+)', block)
        row['Median Survival Steps'] = extract(r'Survival Steps \(Depth\) - Median:\s+([0-9.]+)', block)

        # 3. 提取输入级多样性 (Input Diversity)
        input_match = re.search(r'\[3\. Input Diversity.*?Diversity AUC.*?:\s+[0-9.]+', block, re.DOTALL)
        if input_match:
            in_text = input_match.group(0)
            row['Input K*'] = extract(r'Clusters Discovered \(K\*\):\s+(\d+)', in_text, True)
            row['Input Silhouette'] = extract(r'Absolute Silhouette Score:\s+([+-]?[0-9.]+)', in_text)
            row['Input Intra-Dist'] = extract(r'Avg Intra-Cluster Dist.*:\s+([0-9.]+)', in_text)
            row['Input Inter-Dist'] = extract(r'Avg Inter-Cluster Dist.*:\s+([0-9.]+)', in_text)
            row['Input Entropy'] = extract(r'Entropy.*:\s+([0-9.]+)', in_text)
            row['Input TTD (hrs)'] = extract(r'Mean Time-to-Discovery.*:\s+([0-9.]+)', in_text)
            row['Input AUC'] = extract(r'Diversity AUC.*:\s+([0-9.]+)', in_text)

        # 4. 提取输出级多样性 (Output Diversity)
        output_match = re.search(r'\[3\. Output Diversity.*?Diversity AUC.*?:\s+[0-9.]+', block, re.DOTALL)
        if output_match:
            out_text = output_match.group(0)
            row['Output K*'] = extract(r'Clusters Discovered \(K\*\):\s+(\d+)', out_text, True)
            row['Output Silhouette'] = extract(r'Absolute Silhouette Score:\s+([+-]?[0-9.]+)', out_text)
            row['Output Intra-Dist'] = extract(r'Avg Intra-Cluster Dist.*:\s+([0-9.]+)', out_text)
            row['Output Inter-Dist'] = extract(r'Avg Inter-Cluster Dist.*:\s+([0-9.]+)', out_text)
            row['Output Entropy'] = extract(r'Entropy.*:\s+([0-9.]+)', out_text)
            row['Output TTD (hrs)'] = extract(r'Mean Time-to-Discovery.*:\s+([0-9.]+)', out_text)
            row['Output AUC'] = extract(r'Diversity AUC.*:\s+([0-9.]+)', out_text)

        # 5. 演化深度分析 (Evolutionary Depth)
        row['Avg Crash Gen'] = extract(r'Average Crash Generation \(Mean\):\s+([0-9.]+)', block)
        row['Median Crash Gen'] = extract(r'Median Crash Generation \(Median\):\s+([0-9.]+)', block)
        row['Deepest Crash Gen'] = extract(r'Deepest Crash Found at Generation:\s+(\d+)', block, True)

        results.append(row)

    # 转换为 DataFrame 并导出
    if results:
        df = pd.DataFrame(results)
        # 为方便观察，将浮点数四舍五入到4位
        df = df.round(4)
        df.to_excel(output_excel, index=False)
        print(f"\n✅ 成功解析了 {len(results)} 次运行的评测结果。")
        print(f"✅ 数据已保存至: {output_excel}")
        
        # 在终端打印一个预览
        print("\n--- 数据预览 (部分核心列) ---")
        print(df[['Run ID', 'Hit Ratio (%)', 'Space Coverage (%)', 'Input AUC', 'Output AUC', 'Avg Crash Gen']])
    else:
        print("未提取到任何数据，请检查日志文件的格式。")

if __name__ == "__main__":
    # 配置输入日志和输出 Excel 的路径
    INPUT_LOG = "raw_results.txt"
    OUTPUT_EXCEL = "fuzzing_metrics_summary.xlsx"
    
    parse_fuzzing_logs(INPUT_LOG, OUTPUT_EXCEL)