import os
import torch
from mc_utils import load_model
from mc_framework import MAPElitesFramework

if __name__ == '__main__':
    # 限制线程
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    # --- 路径配置 ---
    # 当前脚本目录: .../RL_MountainCar_Framework
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型路径: .../RL_MountainCar/logs/dqn/MountainCar-v0_8/best_model.zip
    model_path = os.path.join(base_dir, '..', 'RL_MountainCar', 'logs', 'dqn', 'MountainCar-v0_8', 'best_model.zip')
    model_path = os.path.abspath(model_path) # 解析 .. 获取绝对路径
    
    results_dir = os.path.join(base_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print(f"Loading model from: {model_path}")

    # --- 实验参数 ---
    model = load_model(model_path)
    
    # 12小时测试
    time_budget_hours = 12
    init_budget = 10000
    
    print(f"--- Running MAP-Elites for MountainCar ({time_budget_hours} hours) ---")
    
    f = MAPElitesFramework(
        rand_seed=0, 
        cell_granularity=50, 
        descriptors=[0, 1] # Pos, Vel
    )
    
    f.test_policy(
        model=model, 
        env_seed=0, 
        time_budget_hours=time_budget_hours, 
        init_budget=init_budget, 
        results_fp=os.path.join(results_dir, 'mc_test')
    )