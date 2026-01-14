import pickle
import numpy as np
import time
import argparse
import sys
import os
sys.path.append(os.getcwd())
from utils import create_test_env

def teleport_robot(env, state_dict):
    """
    核心传送函数：将机器人强制设置到指定的物理状态
    env: 必须是底层的 Box2D 环境
    """
    base_env = env.unwrapped
    # 1. 恢复躯干 (Hull)
    h = base_env.hull
    h.position = state_dict["hull_pos"]
    h.angle = state_dict["hull_angle"]
    h.linearVelocity = state_dict["hull_lin_vel"]
    h.angularVelocity = state_dict["hull_ang_vel"]
    h.awake = True 
    
    # 2. 恢复腿部 (Legs)
    for i, leg in enumerate(base_env.legs):
        if i < len(state_dict["legs"]):
            s = state_dict["legs"][i]
            leg.position = s["pos"]
            leg.angle = s["angle"]
            leg.linearVelocity = s["lin_vel"]
            leg.angularVelocity = s["ang_vel"]
            leg.awake = True         
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="physics_trajectory.pkl", help="Path to physics trajectory file")
    parser.add_argument("--env", type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument("--steps-before-crash", type=int, default=60, help="Teleport to N steps before the end of trajectory")
    parser.add_argument("--num-cases", type=int, default=5, help="How many crashes to verify")
    parser.add_argument("--seed", type=int, default=0) 
    parser.add_argument("--norm-reward", action="store_true", default=False)
    
    args = parser.parse_args()

    #1. 加载数据
    print(f"Loading trajectories from {args.file}...")
    with open(args.file, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} trajectories.")

    if len(data) == 0:
        print("Data is empty.")
        return

    # 2. 创建环境
    print(f"Creating environment {args.env} using create_test_env...")
    vec_env = create_test_env(
        args.env,
        n_envs=1, 
        stats_path=None,
        seed=args.seed,
        log_dir=None,
        should_render=True, 
        hyperparams={},
        env_kwargs={}
    )
    raw_env = vec_env.envs[0].unwrapped
    print(f"Working with raw environment: {type(raw_env)}")

    # 3. 循环验证
    num_to_verify = min(args.num_cases, len(data))
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    selected_indices = indices[:num_to_verify]

    for i, idx in enumerate(selected_indices):
        case = data[idx]
        seed_data = case['seed'] 
        traj = case['trajectory']
        
        print(f"\n[Case {i+1}/{num_to_verify}] Mutate States (Seed): {seed_data}")
        print(f"Trajectory Length: {len(traj)}")
        
        start_idx = max(0, len(traj) - args.steps_before_crash)
        target_state = traj[start_idx]
        
        print(f"Teleporting to step {start_idx} (Crash happens at {len(traj)})...")
        # A. 重置环境
        raw_env.reset(states=seed_data)

        # B. 传送机器人
        success = teleport_robot(raw_env, target_state)

        # C. 初始渲染
        raw_env.render()
        #初始帧展示时间
        time.sleep(1) 
        
        # D. 模拟运行
        print("Simulating...")
        for _ in range(60):
            action = raw_env.action_space.sample()
            raw_env.step(action)
            raw_env.render()
            time.sleep(0.02) 
        print("Done. Next case...")
        time.sleep(0.5)
    if hasattr(vec_env, 'close'):
        vec_env.close()
    else:
        raw_env.close()

if __name__ == "__main__":
    main()