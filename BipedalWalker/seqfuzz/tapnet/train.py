from __future__ import division
from __future__ import print_function

import math
import sys
import time
import argparse

import torch.optim as optim
from models import TapNet
from tapnet.contrastive import ContrastiveLoss
from tapnet.read_data import get_data_siamese, get_data_siamese2
from utils import *
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()

# dataset settings
parser.add_argument('--data_path', type=str, default="./data/",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="NATOPS", #NATOPS
                    help='time series dataset. Options: See the datasets list')

# cuda settings
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# Training parameter settings
parser.add_argument('--epochs', type=int, default=40,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='Initial learning rate. default:[0.00001]')
parser.add_argument('--wd', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters). default: 5e-3')
parser.add_argument('--stop_thres', type=float, default=1e-9,
                    help='The stop threshold for the training error. If the difference between training losses '
                         'between epoches are less than the threshold, the training will be stopped. Default:1e-9')

# Model parameters


parser.add_argument('--use_cnn', type=boolean_string, default=True,
                    help='whether to use CNN for feature extraction. Default:False')
parser.add_argument('--use_lstm', type=boolean_string, default=True,
                    help='whether to use LSTM for feature extraction. Default:False')
parser.add_argument('--use_rp', type=boolean_string, default=True,
                    help='Whether to use random projection')
parser.add_argument('--rp_params', type=str, default='-1,3',
                    help='Parameters for random projection: number of random projection, '
                         'sub-dimension for each random projection')
parser.add_argument('--use_metric', action='store_true', default=False,
                    help='whether to use the metric learning for class representation. Default:False')
parser.add_argument('--metric_param', type=float, default=0.01,
                    help='Metric parameter for prototype distances between classes. Default:0.000001')
parser.add_argument('--filters', type=str, default="256,256,128",
                    help='filters used for convolutional network. Default:256,256,128')
parser.add_argument('--kernels', type=str, default="8,5,3",
                    help='kernels used for convolutional network. Default:8,5,3')
parser.add_argument('--dilation', type=int, default=1,
                    help='the dilation used for the first convolutional layer. '
                         'If set to -1, use the automatic number. Default:-1')
parser.add_argument('--layers', type=str, default="500,300",
                    help='layer settings of mapping function. [Default]: 500,300')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability). Default:0.5')
parser.add_argument('--lstm_dim', type=int, default=256,
                    help='Dimension of LSTM Embedding.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
args.sparse = True
args.layers = [int(l) for l in args.layers.split(",")]
args.kernels = [int(l) for l in args.kernels.split(",")]
args.filters = [int(l) for l in args.filters.split(",")]
args.rp_params = [float(l) for l in args.rp_params.split(",")]

if not args.use_lstm and not args.use_cnn:
    print("Must specify one encoding method: --use_lstm or --use_cnn")
    print("Program Exiting.")
    exit(-1)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


print("Loading dataset", args.dataset, "...")
# Model and optimizer
model_type = "TapNet" 

if model_type == "TapNet":

    features, labels, idx_train, idx_val, idx_test, nclass = load_raw_ts(args.data_path, dataset=args.dataset)


    # update random permutation parameter
    if args.rp_params[0] < 0:
        dim = features.shape[1]
        args.rp_params = [3, math.floor(dim / (3 / 2))]
    else:
        dim = features.shape[1]
        args.rp_params[1] = math.floor(dim / args.rp_params[1])
    
    args.rp_params = [int(l) for l in args.rp_params]
    print("rp_params:", args.rp_params)

    # update dilation parameter
    if args.dilation == -1:
        args.dilation = math.floor(features.shape[2] / 64)

    print("Data shape:", features.size())
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
   
    # cuda
    if args.cuda:
        #model = nn.DataParallel(model) Used when you have more than one GPU. Sometimes work but not stable
        model.cuda()
        features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()
    input = (features, labels, idx_train, idx_val, idx_test)

# init the optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.wd)


siamese_train_p1, siamese_train_p2, siamese_test_p1, siamese_test_p2, labels_train, labels_test = get_data_siamese2(features, labels, idx_train, idx_val, idx_test)

criterion = torch.nn.BCEWithLogitsLoss()

def handlePred(pre):
    pred2 = []
    for i in range(len(pre)):
        if pre[i][0] > pre[i][1] * 200:
            pred2.append(0)
        else:
            pred2.append(1)
    return pred2

def output_txt(pre, label):
    print(pre[0:5])
    pred_list = pre.tolist()  # 239 * 2
    preds = handlePred(pred_list)
    labels = label.cpu().numpy()
    filepath1 = 'data/result/preds.txt'
    filepath2 = 'data/result/labels.txt'

    for i in range(len(preds)):
        data = str(preds[i]) + '\n'
        f = open(filepath1, 'a', encoding='utf-8')
        f.write(data)
        f.close

    for i in range(len(labels)):
        data = str(labels[i]) + '\n'
        f = open(filepath2, 'a', encoding='utf-8')
        f.write(data)
        f.close


def output_siamese(pre, label):
    filepath1 = 'data/result/sia_preds.txt'
    filepath2 = 'data/result/sia_labels.txt'

    for i in range(len(pre)):
        data = str(pre[i]) + '\n'
        f = open(filepath1, 'a', encoding='utf-8')
        f.write(data)
        f.close

    for i in range(len(label)):
        data = str(label[i]) + '\n'
        f = open(filepath2, 'a', encoding='utf-8')
        f.write(data)
        f.close


# training function
def train():
    loss_list = [sys.maxsize]
    test_best_possible, best_so_far = 0.0, sys.maxsize
    input = (features, labels, idx_train, idx_val, idx_test)

    batch_size = 128
    all_num = (int)(len(siamese_train_p1) / batch_size)

    loss = 0

    for epoch in range(args.epochs):
        for i in range(all_num):
            siameseP1 = siamese_train_p1[i * batch_size : (i + 1) * batch_size]
            siameseP2 = siamese_train_p2[i * batch_size : (i + 1) * batch_size]
            label = labels_train[i * batch_size: (i + 1) * batch_size]

            model.train()
            optimizer.zero_grad()
            # features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()
            siameseP1 = torch.FloatTensor(np.array(siameseP1)).cuda()
            siameseP2 = torch.FloatTensor(np.array(siameseP2)).cuda()
            label = torch.FloatTensor(label).cuda()
            label = label.unsqueeze(1)
            # output = model(input)
            output = model(siameseP1, siameseP2)
            loss_train = criterion(output, label)

            loss = loss_train.item()

            if abs(loss_train.item() - loss_list[-1]) < args.stop_thres \
                    or loss_train.item() > loss_list[-1]:
                break
            else:
                loss_list.append(loss_train.item())

            loss_train.backward()
            optimizer.step()

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.8f}'.format(loss))
    torch.save(model.state_dict(), './data/weights/tapnet.pkl')


def trainTap():
    loss_list = [sys.maxsize]
    test_best_possible, best_so_far = 0.0, sys.maxsize
    input = (features, labels, idx_train, idx_val, idx_test)

    batch_size = 128
    all_num = (int)(len(siamese_train_p1) / batch_size)

    loss = 0
    model.train()
    for epoch in range(args.epochs):
        for i in range(all_num):
            siameseP1 = siamese_train_p1[i * batch_size : (i + 1) * batch_size]
            siameseP2 = siamese_train_p2[i * batch_size : (i + 1) * batch_size]
            label = labels_train[i * batch_size: (i + 1) * batch_size]

            optimizer.zero_grad()
            # features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()
            siameseP1 = torch.FloatTensor(np.array(siameseP1)).cuda()
            siameseP2 = torch.FloatTensor(np.array(siameseP2)).cuda()
            label = torch.FloatTensor(label).cuda()
            label = label.unsqueeze(1)
            # output = model(input)
            output = model(siameseP1, siameseP2)
            loss_train = criterion(output, label)

            loss = loss_train.item()

            if abs(loss_train.item() - loss_list[-1]) < args.stop_thres \
                    or loss_train.item() > loss_list[-1]:
                break
            else:
                loss_list.append(loss_train.item())

            loss_train.backward()
            optimizer.step()

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.8f}'.format(loss))
    torch.save(model.state_dict(), './data/weights/tapnet.pkl')


# test function
def test():
    batch_size = 128
    all_num = (int)(len(siamese_test_p1) / batch_size)

    truth = []
    preds = []

    model.load_state_dict(torch.load(r'./data/weights/tapnet.pkl'))

    model.eval()

    # with torch.no_grad():
    for i in range(all_num):
        # print(i)
        siameseP1 = siamese_test_p1[i * batch_size: (i + 1) * batch_size]
        siameseP2 = siamese_test_p2[i * batch_size: (i + 1) * batch_size]
        label = labels_test[i * batch_size: (i + 1) * batch_size]

        siameseP1 = torch.FloatTensor(np.array(siameseP1)).cuda()
        siameseP2 = torch.FloatTensor(np.array(siameseP2)).cuda()
        label = torch.FloatTensor(label).cuda()
        label = label.unsqueeze(1)

        output = model(siameseP1, siameseP2)
        output = torch.nn.Sigmoid()(output)

        label = label.cpu()
        output = output.cpu()
        for l in label:
            truth.append(l[0])
        for out in output:
            if out[0] > 0.4:
                preds.append(1)
            else:
                preds.append(0)

        # print('*'*20)
        # print(i)
        # print(output)

    precision = precision_score(truth, preds)
    recall = recall_score(truth, preds)
    f1 = f1_score(truth, preds)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)

    # pred = output[idx_test]
    # truth = labels[idx_test]
    output_siamese(preds, truth)

# Train model
t_total = time.time()
trainTap()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
