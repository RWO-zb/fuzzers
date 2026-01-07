import os
import torch
from framework import MAPElitesFramework
from carla_common import load_model

if __name__ == '__main__':
    # Settings
    main_seed = 2024
    env_seed = 2024
    
    # 2 hours test
    time_budget_hours = 2.0 
    init_budget = 50 # Initial random samples
    cell_granularity = 50
    
    # Dummy descriptors (logic handled inside carla_common/framework)
    descriptors = [0, 1] 

    results_fp = 'results_carla_qd'
    if not os.path.isdir(results_fp):
        os.mkdir(results_fp)

    print(f"--- Running MAP-Elites on CARLA for {time_budget_hours} hours ---")
    
    # Init Env
    env_manager = load_model()
    
    # Init Framework
    f = MAPElitesFramework(main_seed, cell_granularity, descriptors=descriptors, name='MAP-Elites-CARLA')
    
    # Run
    f.test_policy(env_manager, env_seed, time_budget_hours, init_budget, results_fp)