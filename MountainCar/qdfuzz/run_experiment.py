import os
import torch
from mc_utils import load_model
from mc_framework import MAPElitesFramework

if __name__ == '__main__':
    # --- 1. 环境配置 ---
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    # --- 2. 路径配置 ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 自动定位到上级目录的 logs 文件夹
    model_path = os.path.join(base_dir, '..', 'logs', 'dqn', 'MountainCar-v0_8', 'best_model.zip')
    model_path = os.path.abspath(model_path)
    
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at: {model_path}")
        exit(1)
        
    results_dir = os.path.join(base_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print(f"Loading model from: {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        exit(1)
    
    # --- 3. 实验预算设置 (关键修改) ---
    # 您可以在这里灵活配置：
    # 模式 A: 仅使用时间 (运行 12 小时)
    # TIME_BUDGET_HOURS = 12
    # SAMPLE_BUDGET = None
    
    # 模式 B: 仅使用样本数 (运行 100,000 次 Oracle 调用)
    # TIME_BUDGET_HOURS = None
    # SAMPLE_BUDGET = 100000

    # 模式 C: 混合模式 (哪个先到就停哪个)
    TIME_BUDGET_HOURS = None       # 例如：最长跑 12 小时
    SAMPLE_BUDGET = 7000       # 例如：最多跑 50万 个样本
    
    INIT_BUDGET = 2000          # 初始化阶段尝试的样本数 (包含在总 Sample Budget 内)

    print(f"--- Running MAP-Elites for MountainCar ---")
    print(f"Configuration: Time Limit={TIME_BUDGET_HOURS}h, Sample Limit={SAMPLE_BUDGET}")
    
    f = MAPElitesFramework(
        rand_seed=42, 
        cell_granularity=50, 
        descriptors=[0, 1] 
    )
    
    f.test_policy(
        model=model, 
        env_seed=42, 
        results_fp=os.path.join(results_dir, 'mc_test'),
        init_budget=INIT_BUDGET,
        time_budget_hours=TIME_BUDGET_HOURS,  # 传入时间预算
        max_samples=SAMPLE_BUDGET             # 传入样本预算
    )