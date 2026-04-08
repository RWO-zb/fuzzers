import pandas as pd
import numpy as np

def parse_to_seconds(time_entry):
    """将 H:M:S 或 纯小时数字 转换为总秒数"""
    if isinstance(time_entry, str) and ':' in time_entry:
        parts = list(map(int, time_entry.split(':')))
        if len(parts) == 3: # H:M:S
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2: # M:S
            return parts[0] * 60 + parts[1]
    try:
        # 纯数字（如 12）按小时计算
        return float(time_entry) * 3600
    except (ValueError, TypeError):
        return 0.0

# --- 完整的 90 组数据录入 ---
raw_data = {
    "curefuzz": {
        "counts": [782, 780, 797, 784, 766, 49, 100, 105, 38, 46, 8980, 8987, 9025, 8965, 9027],
        "times": ["7:42:12", "8:09:02", "7:57:04", "7:34:20", "7:32:18", "12", "12", "12", "12", "12", "0:56:28", "0:58:51", "0:56:27", "0:57:49", "0:57:07"]
    },
    "g-model": {
        "counts": [740, 801, 751, 805, 772, 125, 105, 93, 77, 72, 43242, 43530, 43731, 42308, 41588],
        "times": ["12"] * 15
    },
    "mdpfuzz": {
        "counts": [8824, 9209, 9054, 8736, 8917, 500, 765, 275, 1294, 437, 178323, 178795, 173822, 179248, 180154],
        "times": ["12"] * 15
    },
    "qdfuzz": {
        "counts": [2458, 2322, 2410, 2604, 2300, 243, 76, 61, 199, 185, 163912, 165449, 166116, 163134, 165455],
        "times": ["12"] * 15
    },
    "seqfuzz": {
        "counts": [71, 87, 91, 99, 81, 55, 64, 51, 32, 42, 8962, 8956, 8996, 9000, 8968],
        "times": ["3:27:49", "4:52:28", "4:49:45", "4:41:31", "4:40:43", "12", "12", "12", "12", "12", "5:55:06", "5:54:00", "5:58:40", "6:04:46", "6:00:22"]
    },
    "random": {
        "counts": [4448, 4456, 4603, 4223, 4267, 130, 151, 143, 144, 140, 168941, 169329, 168952, 169190, 168931],
        "times": ["12:12:00", "12:14:00", "12:16:00", "12:18:00", "12:20:00", "12", "12", "12", "12", "12", "12:12:00", "12:14:00", "12:16:00", "12:18:00", "12:20:00"]
    }
}

all_entries = []

for tool, data in raw_data.items():
    counts = data["counts"]
    times = data["times"]
    
    for i in range(15):
        total_sec = parse_to_seconds(times[i])
        crash_num = counts[i]
        
        # 计算秒数/次
        seconds_per_crash = total_sec / crash_num if crash_num > 0 else 0.0
        
        all_entries.append({
            "Tool": tool,
            "Exp_ID": i + 1,
            "Crashes": crash_num,
            "Total_Sec": total_sec,
            "Sec_Per_Crash": round(seconds_per_crash, 4)
        })

# 转换为 DataFrame
df_full = pd.DataFrame(all_entries)

# --- 设置 Pandas 显示选项以确保打印所有行 ---
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.expand_frame_repr', False) # 不换行显示

print("=== 90组实验详细计算结果 (Seconds Per Crash) ===")
print(df_full.to_string(index=False))

# 如果需要将这 90 个数据直接导出到 Excel 方便你写论文，可以使用下面这一行：
df_full.to_excel("fuzzing_90_results.xlsx", index=False)