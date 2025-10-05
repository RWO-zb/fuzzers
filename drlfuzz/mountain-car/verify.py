import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN


# ======================
# 配置路径
# ======================
modelPath ="D:\\code\\fuzzers\\data\\models\\90.zip"
fixedModelPath = "D:\\code\\fuzzers\\data\\models\\90_repaired.zip"
casePath =  "D:\\code\\fuzzers\\data\\result\\result_DRLFuzz.txt"


# ======================
# 测试函数：运行一个 episode
# ======================
def test(env, model, init_state=None, seed=None):
    obs, info = env.reset(seed=seed)

    if init_state is not None:
        env.env.env.state = np.array(init_state, dtype=np.float32)
        obs = env.env.env.state

    total_reward = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    return total_reward


# ======================
# 验证函数
# ======================
def verify(model, env, num=100):
    scores = []

    # 1. bad cases
    if os.path.exists(casePath):
        cases = np.loadtxt(casePath)
        if cases.ndim == 1:
            cases = [cases]
        print(f"Testing on {len(cases)} bad cases...")
        for c in cases:
            score = test(env, model, init_state=(c[0], c[1]))
            scores.append(score)
        scores = np.array(scores)
        #print("Bad Cases scores:", scores.tolist())
        print(
            "Bad Cases mean:{:.2f} max:{:.2f} min:{:.2f} std:{:.2f}".format(
                np.mean(scores), np.max(scores), np.min(scores), np.std(scores)
            )
        )

    # 2. random cases
    scores = []
    print(f"Testing on {num} random cases...")
    for i in range(num):
        score = test(env, model, seed=i)
        scores.append(score)
    scores = np.array(scores)
    #print("Random Cases scores:", scores.tolist())
    print(
        "Random Cases mean:{:.2f} max:{:.2f} min:{:.2f} std:{:.2f}".format(
            np.mean(scores), np.max(scores), np.min(scores), np.std(scores)
        )
    )


# ======================
# 主入口
# ======================
if __name__ == "__main__":
    env = gym.make("MountainCar-v0")

    print("Before retraining")
    net = DQN.load(modelPath, env=env)
    verify(net, env, num=50)

    print("\nAfter retraining")
    if os.path.exists(fixedModelPath):
        net_fixed = DQN.load(fixedModelPath, env=env)
        verify(net_fixed, env, num=50)
    else:
        print("No repaired model found, skipping...")
