import datetime
import numpy as np
# [修改] 使用相对导入，确保在任何 sys.path 配置下都能找到同级模块
from .Hyperparameter import Hyperparameter


def read_data_tapnet(file_path):
    f = open(file_path, encoding='UTF-8')
    lines = f.readlines()
    data = []
    curSeq = []
    for line in lines:
        d = line.rstrip('\n')
        if d == "######":
            copy = []
            for k in curSeq:
                copy.append(k)
            data.append(copy)
            curSeq = []
        else:
            arr1 = d [1:len(d) - 2].split(' ')
            arr = []
            for i in range(len(arr1)):
                if arr1[i] != '':
                    arr.append(float(arr1[i]))
            curSeq.append(arr)
    ret = []
    for s in data:
        var = []
        for i in range(Hyperparameter.Dimension):
            var.append([])
        for i in range(Hyperparameter.Step):
            for wd in range(Hyperparameter.Dimension):
                var[wd].append(s[i][wd])
        a = []
        for wd in range(Hyperparameter.Dimension):
            a.append(var[wd])
        ret.append(a)
    return ret

def get_data():
    # [注意] 这里的路径可能需要根据实际运行目录调整，或者使用绝对路径
    # 如果运行脚本在 seqfuzz 目录下，这个相对路径通常是有效的，但在 analysis 内部调用时需注意
    failObs_path = './seqfuzz/analysis/tapnet/data/crashStateSeqV2.txt'
    successObs_path = './seqfuzz/analysis/tapnet/data/noCrashStateSeqV2.txt'
    
    # 简单的路径回退尝试，防止路径错误
    import os
    if not os.path.exists(failObs_path):
         # 尝试备用路径 (假设当前工作目录是 seqfuzz)
         failObs_path = './analysis/tapnet/data/crashStateSeqV2.txt'
         successObs_path = './analysis/tapnet/data/noCrashStateSeqV2.txt'

    starttime = datetime.datetime.now()
    # 增加异常处理防止文件不存在导致崩溃
    try:
        failObs_data = read_data_tapnet(failObs_path)
        successObs_data = read_data_tapnet(successObs_path)
    except FileNotFoundError:
        print(f"[WARNING] Data files not found at {failObs_path}. Using empty data.")
        failObs_data = []
        successObs_data = []

    endtime = datetime.datetime.now()
    print('load txt data finished, use time(s): ', (endtime - starttime).seconds)

    return failObs_data, successObs_data


def get_data_siamese(x, labels, idx_train, idx_val, idx_test):
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
    # len0 = len(labels_train)
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
    # len0 = len(labels_train)
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
    if len(noCrash_train) > 0:
        bench_noCrash = noCrash_train[0]
    else:
        # Fallback if data is empty
        bench_noCrash = [0] * Hyperparameter.Step

    if len(crash_train) > 0:
        bench_crash = crash_train[0]
    else:
        bench_crash = [1] * Hyperparameter.Step

    print('bench_noCrash: ')
    print(bench_noCrash)
    print('bench_crash: ')
    print(bench_crash)

    for i in range(len(crash_test)):
        siamese_test_p1.append(crash_test[i])
        siamese_test_p2.append(bench_noCrash)
        labels_test.append(0)
    for i in range(len(crash_test)):
        siamese_test_p1.append(crash_test[i])
        siamese_test_p2.append(bench_crash)
        labels_test.append(1)
    for i in range(len(noCrash_test)):
        siamese_test_p1.append(noCrash_test[i])
        siamese_test_p2.append(bench_noCrash)
        labels_test.append(1)
    for i in range(len(noCrash_test)):
        siamese_test_p1.append(noCrash_test[i])
        siamese_test_p2.append(bench_crash)
        labels_test.append(0)


    return siamese_train_p1, siamese_train_p2, siamese_test_p1, siamese_test_p2, labels_train, labels_test


def get_test_data(x, labels, idx_train, idx_val, idx_test):
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