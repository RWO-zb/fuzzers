import math
import argparse
import numpy as np
import torch

from tapnet.models import TapNet
from tapnet.utils import *
from tapnet import Hyperparameter

def load_tapnet_mode():
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    args.seed = 0
    np.random.seed(0)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    args.sparse = True
    args.layers = "500,300"
    args.layers = [int(l) for l in args.layers.split(",")]
    
    # --- 修改重点：适配 MountainCar 的低维度 (Dim=2) ---
    # 原代码: args.kernels = "8,5,3"
    # 修改为: "2,1,1" (因为输入维度只有2，卷积核不能超过2)
    args.kernels = "2,1,1" 
    # ------------------------------------------------
    
    args.kernels = [int(l) for l in args.kernels.split(",")]
    args.filters = "256,256,128"
    args.filters = [int(l) for l in args.filters.split(",")]
    args.rp_params = '-1,3'
    args.rp_params = [float(l) for l in args.rp_params.split(",")]

    # update random permutation parameter
    if args.rp_params[0] < 0:
        dim = Hyperparameter.Step
        args.rp_params = [3, math.floor(dim / (3 / 2))]
    else:
        dim = Hyperparameter.Step
        args.rp_params[1] = math.floor(dim / args.rp_params[1])

    args.rp_params = [int(l) for l in args.rp_params]

    args.dilation = 1
    # update dilation parameter
    if args.dilation == -1:
        args.dilation = math.floor(Hyperparameter.Dimension / 64)

    model = TapNet(nfeat=Hyperparameter.Step,
                   len_ts=Hyperparameter.Dimension,
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
        # 注意：这里创建一个形状正确的零列表
        padding = [np.zeros(Hyperparameter.Dimension).tolist()] * (Hyperparameter.Step - len(seq))
        seq = seq + padding
    elif len(seq) > Hyperparameter.Step:
        # 如果序列太长，进行截断
        seq = seq[:Hyperparameter.Step]
    
    siameseP2 = [seq]
    siameseP2 = torch.FloatTensor(np.array(siameseP2))
    if torch.cuda.is_available():
        siameseP2 = siameseP2.cuda()
    
    # 确保输入张量有正确的形状 [batch_size, seq_len, input_size]
    if len(siameseP2.shape) == 2:
        siameseP2 = siameseP2.unsqueeze(0)
    
    output1 = model(bench_noCrash0, siameseP2)
    output1 = torch.nn.Sigmoid()(output1)

    if output1[0][0] > 0.43:
        return 1
    else:
        return 0