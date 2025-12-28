import argparse
import os
import sys
import shutil
import numpy as np
from pathlib import Path

# --- 路径设置 ---
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# 导入 carla_executor_pcla
from carla_executor_pcla import PCLAExecutor, PCLAEnv
from mdpfuzz.mdpfuzz import Fuzzer

def clean_output_dir(out_dir: Path):
    """
    清理输出目录中的 MDPFuzz 状态文件，防止加载旧的输入向量格式。
    保留 videos 文件夹，只删除 log 和 state。
    """
    if not out_dir.exists():
        return
        
    print(f"[Info] Cleaning stale MDPFuzz state files in {out_dir}...")
    for ext in ['*.txt', '*.json', '*.pkl', 'summary.csv']:
        for f in out_dir.glob(ext):
            try:
                f.unlink()
                print(f"  - Deleted {f.name}")
            except Exception as e:
                print(f"  - Failed to delete {f.name}: {e}")

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
    
    parser.add_argument("--out-dir", default="./mdpfuzz_results", help="Output directory")
    
    # MDPFuzz 参数
    parser.add_argument("--k", type=int, default=10, help="Number of GMM components")
    parser.add_argument("--tau", type=float, default=0.01, help="Density threshold")
    parser.add_argument("--gamma", type=float, default=0.1, help="Weight update factor")

    args = parser.parse_args()

    # 0. 准备输出目录并清理旧状态
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # [关键修复] 清理旧数据，防止 StartID 错乱
    clean_output_dir(out_path)
    
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
        out_dir=str(out_path)
    )
    
    # 3. 初始化 Fuzzer
    fuzzer = Fuzzer(
        random_seed=args.seed,
        k=args.k,
        tau=args.tau,
        gamma=args.gamma,
        executor=executor
    )
    
    # 4. 配置预算模式
    kwargs = {}
    budget_arg = args.test_budget  # 默认使用次数

    print("="*40)
    print(f"MDPFuzz Configuration:")
    print(f"  - Town: {args.town}")
    print(f"  - Init Budget (Random): {args.init_budget} runs")
    print(f"    (Note: Time Budget starts AFTER these initial runs)")
    
    if args.time_budget is not None:
        print(f"  - Fuzzing Mode: TIME BUDGET")
        print(f"  - Budget: {args.time_budget} seconds")
        # [关键] 传递参数给 mdpfuzz
        kwargs['test_budget_in_seconds'] = args.time_budget
        budget_arg = 999999999 # 设置一个很大的数字，让时间来控制停止
    else:
        print(f"  - Fuzzing Mode: ITERATION BUDGET")
        print(f"  - Budget: {args.test_budget} iterations")
        budget_arg = args.test_budget

    print("="*40)

    # 5. 开始 Fuzzing
    log_path = out_path / f"fuzz_log_{args.town}"
    
    fuzzer.fuzzing(
        n=args.init_budget,
        policy=None,
        test_budget=budget_arg,
        saving_path=str(log_path),
        **kwargs 
    )
    
    print("[Info] Fuzzing completed.")

if __name__ == "__main__":
    main()