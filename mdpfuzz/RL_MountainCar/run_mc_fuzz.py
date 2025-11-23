import os
import sys
import torch
import numpy as np

# 1. 环境配置
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# 2. 导入模块
from fuzz.mdpfuzz import Fuzzer
from mc_executor import MountainCarExecutor

if __name__ == '__main__':
    # --- A. 参数配置 ---
    k = 5
    tau = 0.01
    gamma = 0.1
    seed = 0
    
    # [修改] 更改为时间预算模式
    # test_budget = 500   # 原：总迭代次数 (已弃用)
    duration_hours = 12
    time_budget = duration_hours * 3600  # 12小时，单位：秒
    
    init_budget = 10000    # 初始采样数
    sim_steps = 200     # 单次模拟步数

    # --- B. 路径设置 ---
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型路径
    model_path = os.path.join(current_dir, "logs", "dqn", "MountainCar-v0_8", "best_model.zip")
    
    # 日志输出路径
    log_dir = os.path.join(current_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 文件名中标记为 12h
    save_path = os.path.join(log_dir, f'MC_DQN_NoCov_{k}_{tau}_{gamma}_{seed}_12h')

    print(f"Model Path: {model_path}")
    print(f"Log Path:   {save_path}")
    print(f"Time Budget: {duration_hours} hours ({time_budget} seconds)")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # --- C. 初始化与执行 ---
    executor = MountainCarExecutor(sim_steps=sim_steps, env_seed=0, model_path=model_path)
    policy = executor.load_policy()
    
    fuzzer = Fuzzer(random_seed=seed, k=k, tau=tau, gamma=gamma, executor=executor)
    
    print("Starting Fuzzing (No Coverage Mode, Time-Based)...")
    
    # 调用 fuzzing_no_coverage 方法
    fuzzer.fuzzing_no_coverage(
        n=init_budget,
        policy=policy,
        time_budget=time_budget, # [修改] 传入时间预算
        saving_path=save_path,
        local_sensitivity=True,
        save_logs_only=False, 
        exp_name='MountainCar DQN (No Cov)'
    )
    
    print("Fuzzing Completed.")