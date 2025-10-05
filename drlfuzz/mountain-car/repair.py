import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv


# ======================
# 路径设置
# ======================
casePath = "D:\\code\\fuzzers\\data\\result\\result_DRLFuzz.txt"
modelPath = "D:\\code\\fuzzers\\data\\models\\90.zip"
savePath = "D:\\code\\fuzzers\\data\\models\\90_repaired.zip"


# ======================
# 加载模型 & 环境
# ======================
env = gym.make("MountainCar-v0")
vec_env = DummyVecEnv([lambda: env])  # SB3 需要 VecEnv
model = DQN.load(modelPath, env=vec_env)


# ======================
# 读取 bad cases
# ======================
def load_cases(path):
    if not os.path.exists(path):
        print("No bad cases file found!")
        return []
    cases = np.loadtxt(path)
    if cases.ndim == 1:
        cases = [cases]
    return [tuple(c) for c in cases]


# ======================
# 生成数据（基于 bad case）
# ======================
def collect_data(cases, episodes=5):
    buffer = ReplayBuffer(
        buffer_size=50000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device,
    )

    for case in cases:
        init_state = np.array(case, dtype=np.float32)
        for _ in range(episodes):
            obs, info = env.reset()
            env.env.env.state = init_state.copy()  # 设置环境状态
            done = False
            truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, done, truncated, info = env.step(action)
                buffer.add(obs, next_obs, action, reward, done or truncated, infos=[info])
                obs = next_obs
    return buffer


# ======================
# 微调模型
# ======================
def repair(lr=1e-4, steps=5000):
    cases = load_cases(casePath)
    if not cases:
        print("No cases to repair, exiting...")
        return
    print(f"Loaded {len(cases)} bad cases")

    buffer = collect_data(cases, episodes=3)
    print(f"Collected {buffer.size()} transitions for fine-tuning")

    # 用收集到的 buffer 来训练
    model.replay_buffer = buffer
    model.learning_rate = lr
    model.learn(total_timesteps=steps, reset_num_timesteps=False)
    model.save(savePath)
    print(f"Repaired model saved to {savePath}")


# ======================
# 主入口
# ======================
if __name__ == "__main__":
    repair(lr=1e-4, steps=10000)
