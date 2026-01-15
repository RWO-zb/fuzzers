import math
import argparse
import numpy as np
import torch

from tapnet.models import TapNet
from tapnet.read_data import get_data_siamese, get_data_siamese2, get_test_data
from tapnet.utils import *

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

from tapnet import Hyperparameter

def load_tapnet_mode():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.seed = 42
    np.random.seed(42)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    args.sparse = True
    args.layers = "500,300"
    args.layers = [int(l) for l in args.layers.split(",")]
    
    # MountainCar 使用小卷积核
    args.kernels = "2,1,1"
    args.kernels = [int(l) for l in args.kernels.split(",")]
    args.filters = "256,256,128"
    args.filters = [int(l) for l in args.filters.split(",")]
    args.rp_params = '-1,3'
    args.rp_params = [float(l) for l in args.rp_params.split(",")]

    # --- 修正 1: dim 应该是特征维度 (Dimension)，而不是时间步长 (Step) ---
    if args.rp_params[0] < 0:
        dim = Hyperparameter.Dimension  # Dimension = 2
        args.rp_params = [3, math.floor(dim / (3 / 2))]
    else:
        dim = Hyperparameter.Dimension  # Dimension = 2
        args.rp_params[1] = math.floor(dim / args.rp_params[1])

    args.rp_params = [int(l) for l in args.rp_params]

    args.dilation = 1
    # update dilation parameter
    if args.dilation == -1:
        args.dilation = math.floor(Hyperparameter.Dimension / 64)

    # --- 修正 2: TapNet 初始化参数互换 ---
    # 确保这里的 nfeat=2, len_ts=80，与训练时保持一致
    model = TapNet(
                   nfeat=Hyperparameter.Dimension,  # 2
                   len_ts=Hyperparameter.Step,      # 80
                   layers=args.layers,
                   nclass=Hyperparameter.nclass,
                   dropout=0,
                   use_lstm=True,
                   use_cnn=True,
                   filters=args.filters,
                   dilation=args.dilation,
                   kernels=args.kernels,
                   use_metric=False,
                   use_rp=True,
                   rp_params=args.rp_params,
                   lstm_dim=256
                   )
    return model

def predict_once(model, bench_noCrash0, seq):
    # 确保 seq 有正确的形状
    if len(seq) < Hyperparameter.Step:
        # 如果序列长度不够，进行填充
        padding = [[0] * Hyperparameter.Dimension] * (Hyperparameter.Step - len(seq))
        seq = seq + padding
    elif len(seq) > Hyperparameter.Step:
        # 如果序列太长，进行截断
        seq = seq[:Hyperparameter.Step]
    
    # seq 此时是 list 形式 [Step, Dim] -> [80, 2]
    
    # 转换为 Tensor: [1, Step, Dim] -> [1, 80, 2]
    siameseP2 = [seq]
    siameseP2 = torch.FloatTensor(np.array(siameseP2))
    
    if torch.cuda.is_available():
        siameseP2 = siameseP2.cuda()
    
    # --- 关键修正：转置输入数据 ---
    # 训练时的输入是 (N, Dim, Step) 即 (N, 2, 80)
    # 当前 siameseP2 是 (N, 80, 2)，需要转置
    if siameseP2.shape[2] == Hyperparameter.Dimension: # 如果最后一维是2
        siameseP2 = siameseP2.transpose(1, 2) # 变为 (1, 2, 80)
    
    # 同时也要检查 bench_noCrash0 并进行转置
    # enjoy.py 中创建的 bench_noCrash 通常也是 (1, 80, 2)
    if bench_noCrash0.shape[2] == Hyperparameter.Dimension:
        bench_noCrash0 = bench_noCrash0.transpose(1, 2)

    # 现在两个输入都是 (1, 2, 80)，符合模型期望
    output1 = model(bench_noCrash0, siameseP2)
    output1 = torch.nn.Sigmoid()(output1)

    if output1[0][0] > 0.43:
        return 1
    else:
        return 0