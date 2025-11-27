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
    
    # --- 适配 MountainCar 的低维度 (Dim=2) ---
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

# --- 新增：K-Voting 预测函数 ---
def predict_voting(model, golden_sequences_tensor, seq, threshold=0.43):
    """
    实现论文中的 K-Voting 机制。
    将当前序列 seq 与一组 golden_sequences_tensor 进行比对。
    """
    # 1. 预处理当前序列 seq (填充或截断)
    if len(seq) < Hyperparameter.Step:
        padding = [np.zeros(Hyperparameter.Dimension).tolist()] * (Hyperparameter.Step - len(seq))
        seq = seq + padding
    elif len(seq) > Hyperparameter.Step:
        seq = seq[:Hyperparameter.Step]
    
    # 2. 将当前序列转换为 Tensor
    current_seq_tensor = torch.FloatTensor(np.array([seq]))
    if torch.cuda.is_available():
        current_seq_tensor = current_seq_tensor.cuda()
    
    # 3. 构造 Batch 输入
    # golden_sequences_tensor 形状为 [K, Step, Dim]
    # 我们需要将 current_seq_tensor 复制 K 次以匹配形状 [K, Step, Dim]
    K = golden_sequences_tensor.size(0)
    current_seq_batch = current_seq_tensor.repeat(K, 1, 1)
    
    # 4. 模型批量预测
    # model.forward 接受两个 [Batch, Step, Dim] 的输入
    with torch.no_grad():
        output = model(golden_sequences_tensor, current_seq_batch)
        probs = torch.nn.Sigmoid()(output) # 形状 [K, 1]
    
    # 5. 投票统计
    # 假设输出 > threshold 表示“相似”（即预测为 Non-Diverse/Class 0 或 1，取决于训练标签定义）
    # 根据代码逻辑，这里 output > threshold 意味着 Similar to Golden (Non-Crash)
    votes = (probs > threshold).sum().item()
    
    # 论文逻辑：如果超过半数认为相似，则判定为 Non-Diverse，需要终止
    if votes > (K / 2):
        return 1 # 终止信号 (Terminate)
    else:
        return 0 # 继续信号 (Continue)

# 保留旧函数以兼容（如果需要）
def predict_once(model, bench_noCrash0, seq):
    return predict_voting(model, bench_noCrash0, seq)