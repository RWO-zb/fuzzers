import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置 ---
LOG_FILE = 'selection_log.pkl'
PLOT_FILE_NAME = 'crashes_over_time.png'
# --- 结束配置 ---

def load_data(file_path):
    """加载 selection_log.pkl 文件"""
    if not os.path.exists(file_path):
        print(f"错误: 未找到文件: {file_path}")
        print("请确保此脚本与 'selection_log.pkl' 位于同一文件夹中。")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"加载 pickle 文件时出错: {e}")
        return None

def calculate_cumulative_crashes(log_data):
    """
    计算随时间累积的崩溃次数。
    """
    print(f"正在处理 {len(log_data)} 条日志条目...")
    
    cumulative_crashes_list = []
    current_crash_count = 0
    
    for i, entry in enumerate(log_data):
        try:
            # 检查 'did_crash' 字段是否为 True
            if entry.get('did_crash', False):
                current_crash_count += 1
            
            # 无论当前迭代是否崩溃，都记录当前的累积总数
            cumulative_crashes_list.append(current_crash_count)
            
            if (i + 1) % 1000 == 0:
                print(f"  ...已处理 {i + 1} 次迭代。")
                
        except Exception as e:
            print(f"处理条目 {i} (seed_id {entry.get('seed_id')}) 时出错: {e}")

    print("累积崩溃次数计算完成。")
    return cumulative_crashes_list

def plot_crash_trend(crash_data):
    """
    使用 matplotlib 绘制崩溃总数随时间变化的图表
    """
    if not crash_data:
        print("没有找到可绘制的数据。")
        return

    print(f"正在绘制 {len(crash_data)} 个数据点...")
    
    # X 轴：Fuzzing 迭代次数（即种子总数量）
    iterations = range(1, len(crash_data) + 1)
    
    # Y 轴：累积的崩溃次数
    total_crashes = crash_data
    
    plt.figure(figsize=(12, 7))
    
    # 绘制线图
    plt.plot(iterations, total_crashes, label='Total Crashes Found', color='red', linewidth=2)
    
    # 填充线下方的区域，使其更美观
    plt.fill_between(iterations, total_crashes, color='red', alpha=0.1)
    
    # 设置图表属性
    plt.title('Total Crashes Found vs. Fuzzing Iterations')
    plt.xlabel('Number of Fuzzing Iterations (Total Seeds Tested)')
    plt.ylabel('Cumulative Number of Crashes')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0) # 确保 Y 轴从 0 开始
    plt.xlim(left=0)   # 确保 X 轴从 0 开始
    
    # 保存图表
    try:
        plt.savefig(PLOT_FILE_NAME)
        print(f"图表已保存到: {PLOT_FILE_NAME}")
    except Exception as e:
        print(f"保存图表时出错: {e}")

def main():
    log_data = load_data(LOG_FILE)
    if log_data:
        crash_trend_data = calculate_cumulative_crashes(log_data)
        plot_crash_trend(crash_trend_data)

if __name__ == "__main__":
    main()