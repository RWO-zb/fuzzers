import datetime
import numpy as np
import os
from tapnet import Hyperparameter
import random
def read_data_tapnet(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []

    f = open(file_path, encoding='UTF-8')
    lines = f.readlines()
    data = []
    curSeq = []
    
    for line in lines:
        d = line.strip() # 去除换行符和首尾空格
        if not d: continue # 跳过空行

        if d == "######":
            # 一个序列结束
            if len(curSeq) > 0:
                data.append(curSeq)
            curSeq = []
        else:
            # --- 修复部分：适配 'val1, val2, ' 的格式 ---
            # 1. 按逗号分割
            parts = d.split(',')
            arr = []
            for p in parts:
                p = p.strip() # 去除每个数值旁的空格
                if p: # 确保不是空字符串（因为末尾可能有逗号导致最后一个元素为空）
                    try:
                        arr.append(float(p))
                    except ValueError:
                        print(f"Warning: Could not parse '{p}' in line: {d}")
                        pass
            
            # 只有当这一行解析出数据才加入
            if len(arr) > 0:
                curSeq.append(arr)
    
    f.close()

    # --- 数据对齐与转置 ---
    # TapNet 需要 [Dim, Step] 的格式，且长度必须固定
    ret = []
    target_len = Hyperparameter.Step
    target_dim = Hyperparameter.Dimension

    for s in data:
        # 1. 长度对齐 (Padding / Truncating)
        current_len = len(s)
        
        # 如果当前序列是空的，跳过
        if current_len == 0: continue

        # 如果序列太短，用最后一步填充 (Padding)
        if current_len < target_len:
            last_frame = s[-1]
            for _ in range(target_len - current_len):
                s.append(last_frame) # 或者填充0: s.append([0.0] * target_dim)
        
        # 如果序列太长，截断 (Truncating)
        elif current_len > target_len:
            s = s[:target_len]

        # 2. 维度检查与转置
        # 原始 s 是 [Step, Dim]，我们需要转置为 [Dim, Step] 以适配 TapNet
        var = []
        for _ in range(target_dim):
            var.append([])
        
        is_valid = True
        for i in range(target_len):
            # 确保每一步的维度都足够 (比如 MountainCar 应该是 2维)
            if len(s[i]) < target_dim:
                is_valid = False
                break
            for wd in range(target_dim):
                var[wd].append(s[i][wd])
        
        if is_valid:
            ret.append(var)

    return ret

def get_data():
    # 请确保您的 txt 文件确实在这个路径下
    failObs_path = './tapnet/data/crashStateSeqV2.txt'
    successObs_path = './tapnet/data/noCrashStateSeqV2.txt'

    print(f"Reading data from:\n  {failObs_path}\n  {successObs_path}")

    starttime = datetime.datetime.now()
    failObs_data = read_data_tapnet(failObs_path)
    successObs_data = read_data_tapnet(successObs_path)
    endtime = datetime.datetime.now()
    
    print('Load txt data finished.')
    print(f"  Crash Samples: {len(failObs_data)}")
    print(f"  NoCrash Samples: {len(successObs_data)}")
    print('  Time used(s): ', (endtime - starttime).seconds)

    return failObs_data, successObs_data


def get_data_siamese(x, labels, idx_train, idx_val, idx_test):
    # 此函数保持原样，未修改
    train = x[idx_train].tolist()
    test = x[idx_test].tolist()
    labels_train = labels[idx_train].tolist()
    labels_test = labels[idx_test].tolist()

    crash_train, noCrash_train, crash_test, noCrash_test = [], [], [], []

    for i in range(len(train)):
        if labels_train[i][0] == 1:
            crash_train.append(train[i])
        else:
            noCrash_train.append(train[i])

    for i in range(len(test)):
        if labels_test[i][0] == 1:
            crash_test.append(test[i])
        else:
            noCrash_test.append(test[i])

    siamese_train_p1, siamese_train_p2, siamese_test_p1, siamese_test_p2, labels_train, labels_test = [], [], [], [], [], []
    # train:
    for i in range(len(crash_train)):
        for j in range(len(noCrash_train)):
            siamese_train_p1.append(crash_train[i])
            siamese_train_p2.append(noCrash_train[j])
            labels_train.append(0)
    len0 = len(labels_train)
    for i in range(len(crash_train)):
        for j in range(i):
            siamese_train_p1.append(crash_train[i])
            siamese_train_p2.append(crash_train[j])
            labels_train.append(1)
    for i in range(len(noCrash_train)):
        for j in range(i):
            siamese_train_p1.append(noCrash_train[i])
            siamese_train_p2.append(noCrash_train[j])
            labels_train.append(1)

    # test:
    for i in range(len(crash_test)):
        for j in range(i):
            siamese_test_p1.append(crash_test[i])
            siamese_test_p2.append(crash_test[j])
            labels_test.append(1)
    for i in range(len(noCrash_test)):
        for j in range(i):
            siamese_test_p1.append(noCrash_test[i])
            siamese_test_p2.append(noCrash_test[j])
            labels_test.append(1)
    for i in range(len(crash_test)):
        for j in range(len(noCrash_test)):
            siamese_test_p1.append(crash_test[i])
            siamese_test_p2.append(noCrash_test[j])
            labels_test.append(0)

    return siamese_train_p1, siamese_train_p2, siamese_test_p1, siamese_test_p2, labels_train, labels_test


def get_data_siamese2(x, labels, idx_train, idx_val, idx_test):
    # 将 tensor 转为 list
    train = x[idx_train].tolist()
    test = x[idx_test].tolist()
    labels_train = labels[idx_train].tolist()
    labels_test = labels[idx_test].tolist()

    crash_train, noCrash_train, crash_test, noCrash_test = [], [], [], []

    # 分离 Crash 和 NoCrash 数据
    for i in range(len(train)):
        if labels_train[i][0] == 1:
            crash_train.append(train[i])
        else:
            noCrash_train.append(train[i])

    for i in range(len(test)):
        if labels_test[i][0] == 1:
            crash_test.append(test[i])
        else:
            noCrash_test.append(test[i])

    siamese_train_p1, siamese_train_p2, siamese_test_p1, siamese_test_p2 = [], [], [], []
    labels_train, labels_test = [], []

    # --- 关键修改：限制采样数量 (Limit Sampling) ---
    # 我们不进行全排列，而是为每个样本随机匹配固定数量的对子
    # 这样可以将数据量控制在可接受范围内 (例如原始数据的 20 倍，而不是 1000 倍)
    
    PAIR_PER_SAMPLE = 10 # 每个样本只生成 10 个正对和 10 个负对

    # 1. 构建训练集 (Train)
    
    # (A) 负样本对 (Different Class): Crash vs NoCrash
    # 遍历所有 Crash，从 NoCrash 中随机选一些配对
    for c_item in crash_train:
        # 如果 NoCrash 足够多，随机选 PAIR_PER_SAMPLE 个；否则全选
        samples = random.sample(noCrash_train, min(len(noCrash_train), PAIR_PER_SAMPLE))
        for nc_item in samples:
            siamese_train_p1.append(c_item)
            siamese_train_p2.append(nc_item)
            labels_train.append(0) # 0 表示不同类

    # (B) 正样本对 (Same Class): Crash vs Crash
    for c_item in crash_train:
        samples = random.sample(crash_train, min(len(crash_train), PAIR_PER_SAMPLE))
        for other_c in samples:
            siamese_train_p1.append(c_item)
            siamese_train_p2.append(other_c)
            labels_train.append(1) # 1 表示同类

    # (C) 正样本对 (Same Class): NoCrash vs NoCrash
    # NoCrash 数量较多，我们也限制一下采样，避免正样本过多导致不平衡
    # 我们只遍历一部分 NoCrash，或者减少每个 NoCrash 的配对数
    target_nocrash_count = len(crash_train) * 2 # 保持一定比例
    # 随机选取一部分 NoCrash 作为锚点
    subset_nocrash = random.sample(noCrash_train, min(len(noCrash_train), target_nocrash_count))
    
    for nc_item in subset_nocrash:
        samples = random.sample(noCrash_train, min(len(noCrash_train), PAIR_PER_SAMPLE))
        for other_nc in samples:
            siamese_train_p1.append(nc_item)
            siamese_train_p2.append(other_nc)
            labels_train.append(1)

    print(f"Constructed {len(labels_train)} training pairs (Optimized).")

    # 2. 构建测试集 (Test) - 保持原逻辑或同样优化
    # 为了测试速度，简单构建即可
    if len(noCrash_train) > 0 and len(crash_train) > 0:
        bench_noCrash = noCrash_train[0]
        bench_crash = crash_train[0]

        for i in range(len(crash_test)):
            siamese_test_p1.append(crash_test[i])
            siamese_test_p2.append(bench_noCrash)
            labels_test.append(0) # Crash vs Bench_NoCrash -> Different
            
            siamese_test_p1.append(crash_test[i])
            siamese_test_p2.append(bench_crash)
            labels_test.append(1) # Crash vs Bench_Crash -> Same

        for i in range(len(noCrash_test)):
            siamese_test_p1.append(noCrash_test[i])
            siamese_test_p2.append(bench_noCrash)
            labels_test.append(1) # NoCrash vs Bench_NoCrash -> Same

            siamese_test_p1.append(noCrash_test[i])
            siamese_test_p2.append(bench_crash)
            labels_test.append(0) # NoCrash vs Bench_Crash -> Different
    else:
        print("Warning: Not enough train samples for benchmark.")

    return siamese_train_p1, siamese_train_p2, siamese_test_p1, siamese_test_p2, labels_train, labels_test


def get_test_data(x, labels, idx_train, idx_val, idx_test):
    # 此函数保持原样，未修改
    train = x[idx_train].tolist()
    test = x[idx_test].tolist()
    labels_train = labels[idx_train].tolist()
    labels_test = labels[idx_test].tolist()

    crash_train, noCrash_train, crash_test, noCrash_test = [], [], [], []

    for i in range(len(train)):
        if labels_train[i][0] == 1:
            crash_train.append(train[i])
        else:
            noCrash_train.append(train[i])

    for i in range(len(test)):
        if labels_test[i][0] == 1:
            crash_test.append(test[i])
        else:
            noCrash_test.append(test[i])

    return crash_test, noCrash_test