import os
import argparse
import time
from framework import MAPElitesFramework
from carla_common import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QD Fuzzing for CARLA (Aligned with CURE)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--suite", default="full")
    parser.add_argument("--num_vehicles", type=int, default=30)
    parser.add_argument("--fuzz_hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_budget", type=int, default=100)
    
    args = parser.parse_args()

    # Settings
    cell_granularity = 50
    descriptors = [0, 1] 
    
    # Create Result Directory (Timestamped like CURE)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_fp = os.path.join("results", f"{timestamp}_QD_{args.town}_seed{args.seed}")

    print(f"--- Running MAP-Elites on CARLA for {args.fuzz_hours} hours ---")
    print(f"Results will be saved to: {results_fp}")
    
    # Init Env (Logs initialized inside)
    env_manager = load_model(args, results_fp)
    
    # Init Framework
    f = MAPElitesFramework(args.seed, cell_granularity, descriptors=descriptors, name='MAP-Elites-CARLA')
    
    # Run
    f.test_policy(env_manager, args.seed, args.fuzz_hours, args.init_budget, results_fp)