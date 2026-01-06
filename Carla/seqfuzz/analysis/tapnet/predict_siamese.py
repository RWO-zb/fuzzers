import argparse
import torch
import numpy as np
import math
from .models import TapNet
from .Hyperparameter import Hyperparameter

def load_tapnet_mode():
    class Args:
        pass
    args = Args()
    args.layers = [500, 300]
    args.kernels = [8, 5, 3]
    args.filters = [256, 256, 128]
    # [修改] rp_dim 必须为 33 以匹配权重文件
    args.rp_params = [3, 33]
    args.dilation = 1
    
    # 确保模型初始化参数与 Hyperparameter 一致
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

def predict_one(model, seq_np):
    model.eval()
    
    # [新增] 特征维度填充: 17 -> 33
    # 原始 Fuzzer 生成 17 维，TapNet 权重需要 33 维
    current_feat_dim = seq_np.shape[1]
    target_feat_dim = Hyperparameter.Step # 33
    
    if current_feat_dim < target_feat_dim:
        # 在特征维度(axis 1)进行填充
        pad_width = target_feat_dim - current_feat_dim
        # ((top, bottom), (left, right)) -> ((0,0), (0, 16))
        seq_np = np.pad(seq_np, ((0,0), (0, pad_width)), 'constant')
    
    # 时间步长处理: 截取或填充到 Hyperparameter.Dimension (17)
    target_len = Hyperparameter.Dimension
    current_len = seq_np.shape[0]
    
    if current_len < target_len:
        padding = np.zeros((target_len - current_len, target_feat_dim))
        seq_input = np.vstack((padding, seq_np))
    else:
        seq_input = seq_np[-target_len:, :]
        
    # 转换为 Tensor
    # Input format: (Batch, Features, Time) 
    # seq_input shape is (Time, Features) -> (Features, Time) after transpose
    seq_tensor = torch.FloatTensor(seq_input).unsqueeze(0).transpose(1, 2)
    
    # 构造 Benchmark 输入 (使用全0或全1的 Dummy 数据即可，因为只做单次预测)
    # 注意：如果模型 forward 需要 input1, input2，我们需要提供两者
    bench = torch.FloatTensor(np.array([Hyperparameter.bench_noCrash])).transpose(1, 2)
    
    if torch.cuda.is_available():
        seq_tensor = seq_tensor.cuda()
        bench = bench.cuda()
        
    with torch.no_grad():
        # TapNet output is (Batch, 1) -> logit
        output = model(bench, seq_tensor)
        prob = torch.sigmoid(output).item()
        
    # 阈值判断 (假设 > 0.5 为异常)
    return 1 if prob > 0.5 else 0