import pickle
import numpy as np
import os

# --- 配置 ---
# 此脚本假定它被放置在与 'selection_log.pkl' 相同的文件夹中
IN_FILE = 'selection_log.pkl'
OUT_FILE = 'selection_log.txt'
# --- 结束配置 ---

def convert_array_to_string(arr):
    """将 numpy 数组转换为干净的字符串格式, 例如 [1,2,3]"""
    if arr is None:
        return "None"
    try:
        # 将 numpy 数组转换为 python 列表，然后再转换为字符串
        return str(list(arr))
    except Exception:
        # 备用方案，以防它不是一个 numpy 数组
        return str(arr)

def main():
    # 检查输入文件是否存在
    if not os.path.exists(IN_FILE):
        print(f"错误: 未找到文件: {IN_FILE}")
        print("请确保此脚本与 'selection_log.pkl' 位于同一文件夹中。")
        return

    print(f"正在从以下位置加载日志数据: {IN_FILE}")
    try:
        # 加载 pkl 文件
        with open(IN_FILE, 'rb') as f:
            log_data = pickle.load(f)
    except Exception as e:
        print(f"加载 pickle 文件时出错: {e}")
        return

    print(f"正在转换并保存到: {OUT_FILE}")

    # 定义表头。我们使用制表符 ('\t') 作为分隔符。
    headers = [
        "seed_id",
        "parent_id",
        "selection_count",
        "did_crash",
        "seed_state",
        "mutate_state"
    ]
    # 将表头列表连接成一个字符串，并添加换行符
    header_line = "\t".join(headers) + "\n"

    # 打开输出文件
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write(header_line)
        
        # 遍历加载的数据中的每一条日志条目
        for i, entry in enumerate(log_data):
            try:
                # 1. 将每个数据格式化为字符串
                seed_id_str = str(entry.get('seed_id', ''))
                parent_id_str = str(entry.get('parent_id', 'None'))
                selection_count_str = str(entry.get('selection_count', ''))
                did_crash_str = str(entry.get('did_crash', ''))
                
                # 2. 将 state 数组转换为易读的字符串
                seed_state_str = convert_array_to_string(entry.get('seed_state'))
                mutate_state_str = convert_array_to_string(entry.get('mutate_state'))
                
                # 3. 将所有部分组合成一个列表
                line_parts = [
                    seed_id_str,
                    parent_id_str,
                    selection_count_str,
                    did_crash_str,
                    seed_state_str,
                    mutate_state_str
                ]
                
                # 4. 使用制表符连接它们，并写入文件
                line = "\t".join(line_parts) + "\n"
                f.write(line)
            except Exception as e:
                # 处理该行数据时可能发生的任何错误
                print(f"处理条目 {i} (seed_id {entry.get('seed_id')}) 时出错: {e}")

    print("转换完成。")

if __name__ == "__main__":
    main()