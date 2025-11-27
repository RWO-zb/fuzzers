import os
import sys
import torch
import numpy as np

# 1. 环境配置 (保持与 fuzz 脚本一致以确保性能可比性)
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# 2. 导入模块
# 确保可以导入当前目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from fuzz.mdpfuzz import Fuzzer
from mc_executor import MountainCarExecutor

if __name__ == '__main__':
    # --- A. 参数配置 ---
    # 随机测试不需要覆盖率相关的 k, gamma, tau，但初始化 Fuzzer 需要传入
    # 我们可以传入默认值，它们在 random_testing 中不会被使用
    k = 1 
    tau = 0.01
    gamma = 0.1
    seed = 0  # 随机种子
    
    # 测试预算 (Random Testing 通常基于迭代次数)
    # 如果您希望运行大约 12 小时，需要根据单次执行时间估算 n
    # 假设每次执行约 0.05s - 0.1s (取决于模拟步数)，12小时大约是 432,000 - 864,000 次
    # 这里设置一个示例值，您可以根据需要修改
    test_budget = 20000 
    
    sim_steps = 200     # 单次模拟步数

    # --- B. 路径设置 ---
    
    # 模型路径 (指向训练好的 DQN 模型)
    model_path = os.path.join(current_dir, "logs", "dqn", "MountainCar-v0_8", "best_model.zip")
    
    # 日志输出路径
    log_dir = os.path.join(current_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 设置保存路径名称，标记为 RT (Random Testing)
    save_path = os.path.join(log_dir, f'MC_DQN_RT_{seed}_budget{test_budget}')

    print(f"Model Path:  {model_path}")
    print(f"Log Path:    {save_path}")
    print(f"Test Budget: {test_budget} iterations")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # --- C. 初始化与执行 ---
    
    # 1. 初始化执行器
    executor = MountainCarExecutor(sim_steps=sim_steps, env_seed=0, model_path=model_path)
    
    # 2. 加载策略
    policy = executor.load_policy()
    
    # 3. 初始化 Fuzzer
    # 注意：虽然 RT 不用覆盖率，但 Fuzzer 类设计上需要这些参数初始化内部对象
    fuzzer = Fuzzer(random_seed=seed, k=k, tau=tau, gamma=gamma, executor=executor)
    
    print("Starting Random Testing...")
    
    # 4. 执行随机测试
    # check_redundant_input=True 表示如果生成了重复的输入，将跳过执行（不计入 budget）
    fuzzer.random_testing(
        n=test_budget,
        policy=policy,
        path=save_path,
        check_redundant_input=False, 
        exp_name='MountainCar DQN (RT)'
    )
    
    print("Random Testing Completed.")