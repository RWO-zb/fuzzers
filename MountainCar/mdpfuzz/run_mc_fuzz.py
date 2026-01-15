import os
import sys
import torch
import numpy as np

# 1. 环境配置
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# 2. 路径与导入配置 (适应新的文件结构)
current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取 mdpfuzz 文件夹路径
parent_dir = os.path.dirname(current_dir)               # 获取 MountainCar 文件夹路径
sys.path.append(current_dir)                            # 确保能找到 mc_executor
sys.path.append(parent_dir)                             # 备用，确保能找到 logs 等

from fuzz.mdpfuzz import Fuzzer
from mc_executor import MountainCarExecutor

if __name__ == '__main__':
    # --- A. 参数配置 ---
    k = 5
    tau = 0.01
    gamma = 0.1
    seed = 0
    init_budget = 1000   # 初始采样数
    sim_steps = 200      # 单次模拟步数

    # === [关键配置] 预算模式选择 ===
    # 可选值: 'TIME' 或 'ITERATION'
    BUDGET_TYPE = 'ITERATION' 
    
    # 预算具体数值
    BUDGET_HOURS = 12       # 如果选 TIME，运行多少小时
    BUDGET_ITERS = 7000    # 如果选 ITERATION，运行多少次 (不含 init_budget)
    # ============================

    # 计算实际参数
    if BUDGET_TYPE == 'TIME':
        time_budget = BUDGET_HOURS * 3600
        test_budget_val = None
        suffix = f'{BUDGET_HOURS}h'
        print(f"Mode: Time-Based Budget ({BUDGET_HOURS} hours)")
    else:
        time_budget = None
        test_budget_val = BUDGET_ITERS
        suffix = f'{BUDGET_ITERS}it'
        print(f"Mode: Iteration-Based Budget ({BUDGET_ITERS} iterations)")

    # --- B. 路径设置 ---
    # 假设 logs 文件夹在 mdpfuzz 同级 (MountainCar/logs)
    # 也可以设为 mdpfuzz 内部 (MountainCar/mdpfuzz/logs)
    # 这里根据你的截图结构，logs 似乎在 MountainCar/logs
    logs_base_dir = os.path.join(parent_dir, "logs") 
    
    # 模型路径
    model_path = os.path.join(logs_base_dir, "dqn", "MountainCar-v0_8", "best_model.zip")
    
    # 本次运行的日志输出路径
    output_log_dir = os.path.join(current_dir, "logs") # 依然输出到 mdpfuzz/logs 下，方便管理
    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)
        
    save_path = os.path.join(output_log_dir, f'MC_DQN_NoCov_{k}_{tau}_{gamma}_{seed}_{suffix}')

    print(f"Model Path: {model_path}")
    print(f"Log Path:   {save_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        # 尝试备用路径 (如果 logs 在 mdpfuzz 内部)
        model_path_alt = os.path.join(current_dir, "logs", "dqn", "MountainCar-v0_8", "best_model.zip")
        if os.path.exists(model_path_alt):
            print(f"Found model at alternate path: {model_path_alt}")
            model_path = model_path_alt
        else:
            sys.exit(1)

    # --- C. 初始化与执行 ---
    executor = MountainCarExecutor(sim_steps=sim_steps, env_seed=0, model_path=model_path)
    policy = executor.load_policy()
    
    fuzzer = Fuzzer(random_seed=seed, k=k, tau=tau, gamma=gamma, executor=executor)
    
    print("Starting Fuzzing (No Coverage Mode)...")
    
    # 调用 fuzzing_no_coverage 方法
    fuzzer.fuzzing_no_coverage(
        n=init_budget,
        policy=policy,
        test_budget=test_budget_val, # 如果是 Time 模式，这里传 None
        time_budget=time_budget,     # 如果是 Iteration 模式，这里传 None
        saving_path=save_path,
        local_sensitivity=True,
        save_logs_only=False, 
        exp_name='MountainCar DQN (No Cov)'
    )
    
    print("Fuzzing Completed.")