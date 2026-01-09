import argparse
import sys
import time
from pathlib import Path

# --- 路径设置 ---
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from carla_executor_pcla import PCLAExecutor, PCLAEnv
from mdpfuzz.mdpfuzz import Fuzzer

def main():
    parser = argparse.ArgumentParser(description="MDPFuzz / Random Testing with PCLA Agent")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=4000, help="CARLA port")
    parser.add_argument("--town", default="Town01", help="Map name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sim-steps", type=int, default=200, help="Max simulation steps per episode")
    parser.add_argument("--num-vehicles", type=int, default=30, help="Number of NPC vehicles")
    parser.add_argument("--init-budget", type=int, default=100, help="Number of initial random test cases (Only for MDPFuzz)")

    # 互斥组：时间预算 OR 次数预算
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-budget", type=int, default=100, help="Number of fuzzing iterations")
    group.add_argument("--time-budget", type=int, default=None, help="Fuzzing time budget in seconds")
    
    parser.add_argument("--out-dir", default=None, help="Optional override for output directory")
    
    # MDPFuzz 参数
    parser.add_argument("--k", type=int, default=10, help="Number of GMM components")
    parser.add_argument("--tau", type=float, default=0.01, help="Density threshold")
    parser.add_argument("--gamma", type=float, default=0.1, help="Weight update factor")

    parser.add_argument("--method", default="mdpfuzz", choices=["mdpfuzz", "random"], help="Choose fuzzing method")

    args = parser.parse_args()

    # --- 生成输出目录 ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if args.out_dir:
        out_path = Path(args.out_dir)
    else:
        method_suffix = "rt" if args.method == "random" else "mdpfuzz"
        folder_name = f"{timestamp}_{method_suffix}_seed{args.seed}"
        out_path = Path("./results") / folder_name
        
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Results will be saved to: {out_path.resolve()}")

    # 1. 初始化环境
    print(f"[Info] Connecting to CARLA at {args.host}:{args.port}...")
    try:
        env = PCLAEnv(args.host, args.port, args.town, seed=args.seed)
    except Exception as e:
        print(f"[Error] Failed to connect to CARLA: {e}")
        return

    # 2. 初始化 Executor
    executor = PCLAExecutor(
        sim_steps=args.sim_steps,
        env=env,
        num_vehicles=args.num_vehicles,
        out_dir=str(out_path),
        init_budget=args.init_budget
    )
    
    # 3. 初始化 Fuzzer
    fuzzer = Fuzzer(
        random_seed=args.seed,
        k=args.k,
        tau=args.tau,
        gamma=args.gamma,
        executor=executor
    )
    
    # 4. 配置预算模式和 Fuzzing 参数
    kwargs = {}
    budget_arg = args.test_budget
    kwargs['local_sensitivity'] = True # 仅对 MDPFuzz 有效

    print("="*40)
    print(f"Configuration:")
    print(f"  - Method: {args.method.upper()}")
    print(f"  - Town: {args.town}")
    print(f"  - Sim Steps: {args.sim_steps}")
    
    if args.time_budget is not None:
        print(f"  - Budget Mode: TIME")
        print(f"  - Budget: {args.time_budget} seconds")
        kwargs['test_budget_in_seconds'] = args.time_budget
        budget_arg = 999999999 # 时间预算模式下，此处设为极大值以防止迭代次数提前耗尽
    else:
        print(f"  - Budget Mode: ITERATION")
        print(f"  - Budget: {args.test_budget} iterations")
        budget_arg = args.test_budget

    if args.method == "mdpfuzz":
        print(f"  - Init Budget: {args.init_budget}")
        print(f"  - Sensitivity: Local")
    else:
        print(f"  - Strategy: Pure Random Sampling")

    print("="*40)

    # 5. 开始运行
    log_path = out_path / f"mdpfuzz_state"
    
    if args.method == "random":
        print(f"[Info] Starting Random Testing (Baseline)...")
        fuzzer.random_testing(
            n=budget_arg, 
            policy=None,
            path=str(log_path),
            check_redundant_input=False,
            **kwargs
        )
    else:
        print(f"[Info] Starting MDPFuzz...")
        fuzzer.fuzzing_no_coverage(
            n=args.init_budget,
            policy=None,
            test_budget=budget_arg,
            saving_path=str(log_path),
            **kwargs 
        )
    
    print("[Info] Execution completed.")

if __name__ == "__main__":
    main()