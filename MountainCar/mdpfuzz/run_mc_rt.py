import os
import sys
import torch
import numpy as np

# 1. 环境配置
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# 2. 路径与导入配置
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from fuzz.mdpfuzz import Fuzzer
from mc_executor import MountainCarExecutor

if __name__ == '__main__':
    # --- A. 参数配置 ---
    k = 1 
    tau = 0.01
    gamma = 0.1
    seed = 1022
    sim_steps = 200

    # === [关键配置] 预算模式选择 ===
    # 可选值: 'TIME' 或 'ITERATION'
    # RT 默认建议用次数，但如果你想跑固定时间也可以
    BUDGET_TYPE = 'ITERATION' 
    
    BUDGET_HOURS = 12       
    BUDGET_ITERS = 5000 
    # ============================

    if BUDGET_TYPE == 'TIME':
        time_budget = BUDGET_HOURS * 3600
        n_val = None # RT 中 n 用于迭代次数限制
        suffix = f'{BUDGET_HOURS}h'
        print(f"Mode: Time-Based RT ({BUDGET_HOURS} hours)")
    else:
        time_budget = None
        n_val = BUDGET_ITERS
        suffix = f'{BUDGET_ITERS}it'
        print(f"Mode: Iteration-Based RT ({BUDGET_ITERS} iterations)")

    # --- B. 路径设置 ---
    logs_base_dir = os.path.join(parent_dir, "logs")
    model_path = os.path.join(logs_base_dir, "dqn", "MountainCar-v0_8", "best_model.zip")
    
    output_log_dir = os.path.join(current_dir, "logs")
    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)
        
    save_path = os.path.join(output_log_dir, f'MC_DQN_RT_{seed}_{suffix}')

    print(f"Model Path:  {model_path}")
    print(f"Log Path:    {save_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        # 尝试备用路径
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
    
    print("Starting Random Testing...")
    
    fuzzer.random_testing(
        n=n_val,                 # 次数预算
        time_budget=time_budget, # 时间预算
        policy=policy,
        path=save_path,
        check_redundant_input=False, 
        exp_name='MountainCar DQN (RT)'
    )
    
    print("Random Testing Completed.")