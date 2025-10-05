import os
import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from scipy import spatial

# ======================
# 环境 & 模型初始化
# ======================
env = gym.make("MountainCar-v0")
model = DQN.load("D:\\code\\fuzzers\\data\\models\\90.zip")

savePath = "D:\\code\\fuzzers\\data\\result\\result_DRLFuzz.txt"
if os.path.exists(savePath):
    os.remove(savePath)

random.seed(2003511)
np.random.seed(2003511)

allStates = set()
kdTree = None
delta = 0.05   # 状态距离阈值
innerDelta = 0.05
resultNum = []


# ======================
# 测试函数：运行一条轨迹
# ======================
def test(env, model, init_state=None, seed=None):
    """
    运行 MountainCar 环境，返回 episode 累计奖励
    init_state: 如果不为 None，直接把环境置到给定状态
    """
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
# 状态变异函数（简单加噪声）
# ======================
def mutator(state, l):
    pos, vel = state
    pos += np.random.uniform(-l, l)
    vel += np.random.uniform(-l, l)

    # 限制在合法范围
    pos = np.clip(pos, -1.2, 0.6)
    vel = np.clip(vel, -0.07, 0.07)
    return [pos, vel]


# ======================
# 随机生成状态
# ======================
def randFun(coverage=True):
    global delta
    count = 0
    while True:
        pos = random.uniform(-1.2, 0.6)
        vel = random.uniform(-0.07, 0.07)
        count += 1
        if not coverage:
            return [pos, vel]
        if count == 10000:
            delta *= 0.9
        if getDistance((pos, vel)) > delta:
            allStates.add((pos, vel))
            return [pos, vel]


# ======================
# 计算状态与已知状态的最小距离
# ======================
def getDistance(arg):
    if len(allStates) == 0:
        return np.inf
    global kdTree
    kdTree = spatial.KDTree(data=np.array(list(allStates)), leafsize=10000)
    dist, _ = kdTree.query(np.array(list(arg)))
    return dist


# ======================
# DRLFuzz 主循环
# ======================
def DRLFuzz(num, n, l, alpha, theta, coverage):
    statePool = []
    score = []
    resultPool = set()

    # 初始化种子状态
    for _ in range(num):
        s = randFun(coverage)
        statePool.append(s)
        score.append(0)

    # 迭代
    for k in range(n):
        for i in range(num):
            score[i] = test(env, model, init_state=statePool[i])
            if score[i] < theta:
                tmp = tuple(statePool[i])
                if tmp not in resultPool:
                    with open(savePath, "a") as f:
                        f.write(f"{tmp[0]} {tmp[1]}\n")
                resultPool.add(tmp)

        print(f"iteration {k+1}, failed cases num:{len(resultPool)}")
        resultNum.append(len(resultPool))

        # 按分数排序
        idx = sorted(range(len(score)), key=lambda x: score[x])

        for i in range(num):
            if i < int(num * alpha):
                st = mutator(statePool[idx[i]], l)
                if st != statePool[idx[i]]:
                    statePool[idx[i]] = st
                else:
                    statePool[idx[i]] = randFun(coverage)
            else:
                statePool[idx[i]] = randFun(coverage)

    return resultPool


# ======================
# 主程序入口
# ======================
if __name__ == "__main__":
    result = DRLFuzz(
        num=50,     # 种子数
        n=20,       # 迭代次数
        l=0.02,     # 变异幅度
        alpha=0.2,  # top α 部分用变异，其余用随机
        theta=-150, # 判定bad case的奖励阈值
        coverage=True
    )
    print("Fuzzing Done, bad cases:", len(result))
