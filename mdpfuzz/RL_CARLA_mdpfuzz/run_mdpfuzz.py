import argparse
import os
import sys
import shutil
import time
import numpy as np
from pathlib import Path

# --- 路径设置 ---
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# 导入 carla_executor_pcla
from carla_executor_pcla import PCLAExecutor, PCLAEnv
from mdpfuzz.mdpfuzz import Fuzzer

def main():
    parser = argparse.ArgumentParser(description="MDPFuzz with PCLA Agent")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--town", default="Town01", help="Map name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sim-steps", type=int, default=200, help="Max simulation steps per episode")
    
    parser.add_argument("--num-vehicles", type=int, default=10, help="Number of NPC vehicles")
    parser.add_argument("--init-budget", type=int, default=10, help="Number of initial random test cases")

    # 互斥组：时间预算 OR 次数预算
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-budget", type=int, default=100, help="Number of fuzzing iterations")
    group.add_argument("--time-budget", type=int, default=None, help="Fuzzing time budget in seconds")
    
    # 移除默认值，改为在该参数为空时自动生成带时间戳的路径
    parser.add_argument("--out-dir", default=None, help="Optional override for output directory")
    
    # MDPFuzz 参数
    parser.add_argument("--k", type=int, default=10, help="Number of GMM components")
    parser.add_argument("--tau", type=float, default=0.01, help="Density threshold")
    parser.add_argument("--gamma", type=float, default=0.1, help="Weight update factor")

    args = parser.parse_args()

    # --- 生成带时间戳的输出目录 ---
    # 格式模仿 RL_CARLA: results/YYYYMMDD_HHMMSS_mdpfuzz_seedXXX
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if args.out_dir:
        out_path = Path(args.out_dir)
    else:
        # 自动生成路径
        folder_name = f"{timestamp}_mdpfuzz_seed{args.seed}"
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
    # 传入 init_budget 用于判断 Phase 和 命名逻辑
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
    budget_arg = args.test_budget  # 默认使用次数

    # [关键修改 1] 硬编码开启 Local Sensitivity
    kwargs['local_sensitivity'] = True

    print("="*40)
    print(f"MDPFuzz Configuration:")
    print(f"  - Town: {args.town}")
    print(f"  - Init Budget (Random): {args.init_budget} runs")
    print(f"    (Phase 1: seed_000 to seed_{args.init_budget-1:03d})")
    
    if args.time_budget is not None:
        print(f"  - Fuzzing Mode: TIME BUDGET")
        print(f"  - Budget: {args.time_budget} seconds")
        kwargs['test_budget_in_seconds'] = args.time_budget
        budget_arg = 999999999 # 设置极大值，由时间控制退出
    else:
        print(f"  - Fuzzing Mode: ITERATION BUDGET")
        print(f"  - Budget: {args.test_budget} iterations")
        budget_arg = args.test_budget

    # 打印当前配置状态
    print(f"  - Method: Fuzzer (No Coverage, Reward Guided Only)")
    print(f"  - Sensitivity: Local (Math-based, No Extra Sim)")

    print("="*40)

    # 5. 开始 Fuzzing
    # 日志文件保存到同一个 out_path 下
    log_path = out_path / f"mdpfuzz_state"
    
    # [关键修改 2] 使用 fuzzing_no_coverage 替代 fuzzing
    fuzzer.fuzzing_no_coverage(
        n=args.init_budget,
        policy=None,
        test_budget=budget_arg,
        saving_path=str(log_path),
        **kwargs 
    )
    
    print("[Info] Fuzzing completed.")

if __name__ == "__main__":
    main()