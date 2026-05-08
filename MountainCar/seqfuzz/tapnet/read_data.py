import datetime
import numpy as np
import os
from tapnet import Hyperparameter
import random
def read_data_tapnet(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []

    f = open(file_path, encoding='UTF-8')
    lines = f.readlines()
    data = []
    curSeq = []
    
    for line in lines:
        if d == "######":
            if len(curSeq) > 0:
                data.append(curSeq)
            curSeq = []
        else:
            parts = d.split(',')
            arr = []
            for p in parts:
                p = p.strip()
                if p: # Ensure not empty string due to trailing comma
                    try:
                        arr.append(float(p))
                    except ValueError:
                        print(f"Warning: Could not parse '{p}' in line: {d}")
                        pass
            
            if len(arr) > 0:
                curSeq.append(arr)
    
    f.close()

    # --- Data alignment and transposition ---
    # TapNet requires fixed-length [Dim, Step] format
    ret = []
    target_len = Hyperparameter.Step
    target_dim = Hyperparameter.Dimension

    for s in data:
        # 1. Length alignment (Padding / Truncating)
        current_len = len(s)
        
        # Skip empty sequences
        if current_len == 0: continue

        # Pad with last frame if too short
        if current_len < target_len:
            last_frame = s[-1]
            for _ in range(target_len - current_len):
                s.append(last_frame)
        
        # Truncate if too long
        elif current_len > target_len:
            s = s[:target_len]

        # 2. Dimension check and transposition
        # Transpose from [Step, Dim] to [Dim, Step] for TapNet
        var = []
        for _ in range(target_dim):
            var.append([])
        
        is_valid = True
        for i in range(target_len):
            # Ensure correct dimensions (e.g., 2D for MountainCar)
            if len(s[i]) < target_dim:
                is_valid = False
                break
            for wd in range(target_dim):
                var[wd].append(s[i][wd])
        
        if is_valid:
            ret.append(var)

    return ret

def get_data():
    # Paths for data files
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
    # Convert tensor to list
    train = x[idx_train].tolist()
    test = x[idx_test].tolist()
    labels_train = labels[idx_train].tolist()
    labels_test = labels[idx_test].tolist()

    crash_train, noCrash_train, crash_test, noCrash_test = [], [], [], []

    # Separate Crash and NoCrash data
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

    # --- Limit sampling to control data size ---
    # Match each sample with a fixed number of random pairs to keep volume manageable
    
    PAIR_PER_SAMPLE = 10 # Generate 10 positive and 10 negative pairs per sample

    # 1. Construct training set (Train)
    
    # (A) Negative pairs (Different Class): Crash vs NoCrash
    # Iterate through all Crash samples and randomly match with NoCrash samples
    for c_item in crash_train:
        # Select up to PAIR_PER_SAMPLE samples from NoCrash
        samples = random.sample(noCrash_train, min(len(noCrash_train), PAIR_PER_SAMPLE))
        for nc_item in samples:
            siamese_train_p1.append(c_item)
            siamese_train_p2.append(nc_item)
            labels_train.append(0) # 0 indicates different classes

    # (B) Positive pairs (Same Class): Crash vs Crash
    for c_item in crash_train:
        samples = random.sample(crash_train, min(len(crash_train), PAIR_PER_SAMPLE))
        for other_c in samples:
            siamese_train_p1.append(c_item)
            siamese_train_p2.append(other_c)
            labels_train.append(1) # 1 indicates same class

    # (C) Positive pairs (Same Class): NoCrash vs NoCrash
    # Limit NoCrash pairs to prevent class imbalance
    # Subsample NoCrash samples as anchors
    target_nocrash_count = len(crash_train) * 2 # Maintain ratio
    # Randomly select a subset of NoCrash as anchors
    subset_nocrash = random.sample(noCrash_train, min(len(noCrash_train), target_nocrash_count))
    
    for nc_item in subset_nocrash:
        samples = random.sample(noCrash_train, min(len(noCrash_train), PAIR_PER_SAMPLE))
        for other_nc in samples:
            siamese_train_p1.append(nc_item)
            siamese_train_p2.append(other_nc)
            labels_train.append(1)

    print(f"Constructed {len(labels_train)} training pairs (Optimized).")

    # 2. Construct test set (Test)
    # Simplified construction for faster testing
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