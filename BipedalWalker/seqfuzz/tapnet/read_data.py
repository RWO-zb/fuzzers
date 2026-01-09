import datetime
import numpy as np
from tapnet import Hyperparameter


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
    failObs_path = './tapnet/data/crashStateSeqV2.txt'
    successObs_path = './tapnet/data/noCrashStateSeqV2.txt'

    starttime = datetime.datetime.now()
    failObs_data = read_data_tapnet(failObs_path)
    successObs_data = read_data_tapnet(successObs_path)
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

    # siamese_train_p1 = siamese_train_p1[0 : 2 * len0]
    # siamese_train_p2 = siamese_train_p2[0 : 2 * len0]
    # labels_train = labels_train[0 : 2 * len0]

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

    # siamese_train_p1 = siamese_train_p1[0 : 2 * len0]
    # siamese_train_p2 = siamese_train_p2[0 : 2 * len0]
    # labels_train = labels_train[0 : 2 * len0]

    # test:
    bench_noCrash = noCrash_train[0]
    bench_crash = crash_train[0]

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