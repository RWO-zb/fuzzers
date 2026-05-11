import os
import torch
import argparse
from mc_utils import load_model
from mc_framework import MAPElitesFramework

if __name__ == '__main__':
    # --- 0. parser ---
    parser = argparse.ArgumentParser(description="Run MAP-Elites for MountainCar")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for experiment')
    args = parser.parse_args()
    current_seed = args.seed

    # --- 1. Thread configuration (single-threaded for reproducibility) ---
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    # --- 2. Path configuration ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
    
    # --- 3. Budget configuration ---
    # Mode A: Time only          -> TIME_BUDGET_HOURS=12, SAMPLE_BUDGET=None
    # Mode B: Sample count only  -> TIME_BUDGET_HOURS=None, SAMPLE_BUDGET=100000
    # Mode C: Hybrid (whichever limit is reached first)
    TIME_BUDGET_HOURS = 12
    SAMPLE_BUDGET = None
    
    INIT_BUDGET = 10000  # Random samples for the initialization phase (counted in total budget)

    print(f"--- Running MAP-Elites for MountainCar ---")
    print(f"Configuration: Time Limit={TIME_BUDGET_HOURS}h, Sample Limit={SAMPLE_BUDGET}")
    result_prefix = f'mc_test_seed{current_seed}'
    f = MAPElitesFramework(
        rand_seed=current_seed, 
        cell_granularity=50, 
        descriptors=[0, 1] 
    )
    
    f.test_policy(
        model=model, 
        env_seed=current_seed, 
        results_fp=os.path.join(results_dir, result_prefix),
        init_budget=INIT_BUDGET,
        time_budget_hours=TIME_BUDGET_HOURS,
        max_samples=SAMPLE_BUDGET
    )