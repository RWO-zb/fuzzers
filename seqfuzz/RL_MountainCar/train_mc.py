from __future__ import division
from __future__ import print_function

import math
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from tapnet.models import TapNet
from tapnet.read_data import get_data, get_data_siamese2
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# --- 参数设置 ---
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay.')
parser.add_argument('--stop_thres', type=float, default=1e-9, help='Stop threshold.')

# 模型参数 (根据 MountainCar 调整)
parser.add_argument('--use_cnn', type=bool, default=True)
parser.add_argument('--use_lstm', type=bool, default=True)
parser.add_argument('--use_rp', type=bool, default=True)
parser.add_argument('--rp_params', type=str, default='-1,3')
parser.add_argument('--use_metric', action='store_true', default=False)
parser.add_argument('--filters', type=str, default="256,256,128")
parser.add_argument('--kernels', type=str, default="2,1,1") # 关键：MountainCar 维度低，Kernel 必须小
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--layers', type=str, default="500,300")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--lstm_dim', type=int, default=256)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.layers = [int(l) for l in args.layers.split(",")]
args.kernels = [int(l) for l in args.kernels.split(",")]
args.filters = [int(l) for l in args.filters.split(",")]
args.rp_params = [float(l) for l in args.rp_params.split(",")]

print("Loading MountainCar data...")

# 读取数据
crash_list, nocrash_list = get_data()

# 转换为 Numpy 数组
X_crash = np.array(crash_list)
X_nocrash = np.array(nocrash_list)

# 确保数据不为空
if len(X_crash) == 0 or len(X_nocrash) == 0:
    print("Error: 数据不足。请检查 tapnet/data/ 下是否存在非空的 .txt 数据文件。")
    exit(1)

print(f"Crash samples: {len(X_crash)}, NoCrash samples: {len(X_nocrash)}")

# 合并特征
features_np = np.concatenate((X_crash, X_nocrash), axis=0)

# 创建标签 (1: Crash, 0: NoCrash)
labels_crash = np.ones((len(X_crash), 1))
labels_nocrash = np.zeros((len(X_nocrash), 1))
labels_np = np.concatenate((labels_crash, labels_nocrash), axis=0)

# 转换为 Tensor
features = torch.tensor(features_np).float()
labels = torch.tensor(labels_np).long()

# 划分数据集
N = features.shape[0]
indices = np.random.permutation(N)
train_count = int(0.8 * N)

idx_train = torch.tensor(indices[:train_count])
idx_test = torch.tensor(indices[train_count:])
idx_val = idx_test # 暂用测试集作为验证集

nclass = 2

# 适配参数
if args.rp_params[0] < 0:
    dim = features.shape[1]
    args.rp_params = [3, math.floor(dim / (3 / 2))]
else:
    dim = features.shape[1]
    args.rp_params[1] = math.floor(dim / args.rp_params[1])
args.rp_params = [int(l) for l in args.rp_params]

if args.dilation == -1:
    args.dilation = math.floor(features.shape[2] / 64)

print("Layers", args.layers)

# 初始化模型
model = TapNet(nfeat=features.shape[1],
               len_ts=features.shape[2],
               layers=args.layers,
               nclass=nclass,
               dropout=args.dropout,
               use_lstm=args.use_lstm,
               use_cnn=args.use_cnn,
               filters=args.filters,
               dilation=args.dilation,
               kernels=args.kernels,
               use_metric=args.use_metric,
               use_rp=args.use_rp,
               rp_params=args.rp_params,
               lstm_dim=args.lstm_dim
               )

if args.cuda:
    model.cuda()
    # 注意：这里我们把 features 和 labels 放到了 GPU，idx_train 也放到了 GPU
    # 但在传给 get_data_siamese2 时，需要把它们转回 CPU
    features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()

# 准备训练数据
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = torch.nn.BCEWithLogitsLoss()

# --- 关键修改：调用 get_data_siamese2 时，确保所有输入都在 CPU 上 ---
# features.cpu() 和 labels.cpu() 已经在原代码中
# 修改点：增加 idx_train.cpu(), idx_val.cpu(), idx_test.cpu()
siamese_train_p1, siamese_train_p2, siamese_test_p1, siamese_test_p2, labels_train_sia, labels_test_sia = get_data_siamese2(
    features.cpu(), 
    labels.cpu(), 
    idx_train.cpu(), 
    idx_val.cpu(), 
    idx_test.cpu()
)

def trainTap():
    loss_list = [sys.maxsize]
    batch_size = 64
    if len(siamese_train_p1) < batch_size:
        batch_size = len(siamese_train_p1)
    
    all_num = int(len(siamese_train_p1) / batch_size)
    if all_num == 0: all_num = 1

    model.train()
    print("Start Training...")
    for epoch in range(args.epochs):
        loss_val = 0
        for i in range(all_num):
            end_idx = min((i + 1) * batch_size, len(siamese_train_p1))
            start_idx = i * batch_size
            
            sP1 = siamese_train_p1[start_idx : end_idx]
            sP2 = siamese_train_p2[start_idx : end_idx]
            lbl = labels_train_sia[start_idx : end_idx]

            optimizer.zero_grad()
            
            sP1 = torch.FloatTensor(np.array(sP1))
            sP2 = torch.FloatTensor(np.array(sP2))
            lbl = torch.FloatTensor(lbl).unsqueeze(1)
            
            if args.cuda:
                sP1, sP2, lbl = sP1.cuda(), sP2.cuda(), lbl.cuda()

            output = model(sP1, sP2)
            loss_train = criterion(output, lbl)
            loss_val = loss_train.item()

            loss_train.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch: {:04d} | Loss: {:.8f}'.format(epoch + 1, loss_val))
            
    # --- 保存权重 ---
    import os
    save_dir = './tapnet/data/weights/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), save_dir + 'tapnet.pkl')
    print("Model saved to", save_dir + 'tapnet.pkl')

# 开始训练
t_total = time.time()
trainTap()
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))