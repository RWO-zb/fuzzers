import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd
import random
from tapnet import read_data

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def loaddata(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    return a

def generate_train_test(crash_data, noCrash_data):
    noCrash_train_num = 455
    noCrash_test_num = len(noCrash_data) - noCrash_train_num
    crash_train_num = 20
    crash_test_num = len(crash_data) - crash_train_num


    x_train = noCrash_data[0:noCrash_train_num]
    x_train.extend(crash_data[0:crash_train_num])
    y_train = []
    for i in range(noCrash_train_num):
        y_train.append([0])
    for i in range(crash_train_num):
        y_train.append([1])

    x_test = noCrash_data[noCrash_train_num:len(noCrash_data)]
    x_test.extend(crash_data[crash_train_num:len(crash_data)])
    y_test = []
    for i in range(0, noCrash_test_num):
        y_test.append([0])
    for i in range(crash_test_num):
        y_test.append([1])

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    # np.array(ret)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def load_raw_ts(path, dataset, tensor_format=True):

    # path = path + "raw/" + dataset + "/"
    # x_train = np.load(path + 'X_train.npy')
    # y_train = np.load(path + 'y_train.npy')
    # x_test = np.load(path + 'X_test.npy')
    # y_test = np.load(path + 'y_test.npy')

    crash_data, noCrash_data = read_data.get_data()
    x_train, y_train, x_test, y_test = generate_train_test(crash_data, noCrash_data)

    print()
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_test.shape: ', x_test.shape)
    print('y_test.shape: ', y_test.shape)
    print()


    ts = np.concatenate((x_train, x_test), axis=0)
    ts = np.transpose(ts, axes=(0, 2, 1))
    labels = np.concatenate((y_train, y_test), axis=0)
    nclass = int(np.amax(labels)) + 1


    train_size = y_train.shape[0]

    total_size = labels.shape[0]
    idx_train = range(train_size)
    idx_val = range(train_size, total_size)
    idx_test = range(train_size, total_size)

    if tensor_format:
        # features = torch.FloatTensor(np.array(features))
        ts = torch.FloatTensor(np.array(ts))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return ts, labels, idx_train, idx_val, idx_test, nclass

def load_features():

    crash_data, noCrash_data = read_data.get_data()
    x_train, y_train, x_test, y_test = generate_train_test(crash_data, noCrash_data)

    ts = np.concatenate((x_train, x_test), axis=0)
    ts = np.transpose(ts, axes=(0, 2, 1))
    labels = np.concatenate((y_train, y_test), axis=0)
    nclass = int(np.amax(labels)) + 1


    train_size = y_train.shape[0]

    total_size = labels.shape[0]
    idx_train = range(train_size)
    idx_val = range(train_size, total_size)
    idx_test = range(train_size, total_size)

    ts = torch.FloatTensor(np.array(ts))
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return ts, labels, idx_train, idx_val, idx_test, nclass


def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sums = mx.sum(axis=1)
    mx = mx.astype('float32')
    row_sums_inverse = 1 / row_sums
    f = mx.multiply(row_sums_inverse)
    return sp.csr_matrix(f).astype('float32')


def accuracy(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    return accuracy_score



def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output

def dump_embedding(proto_embed, sample_embed, labels, dump_file='./plot/embeddings.txt'):
    proto_embed = proto_embed.cpu().detach().numpy()
    sample_embed = sample_embed.cpu().detach().numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    labels = np.concatenate((np.asarray([i for i in range(nclass)]),
                             labels.squeeze().cpu().detach().numpy()), axis=0)

    with open(dump_file, 'w') as f:
        for i in range(len(embed)):
            label = str(labels[i])
            line = label + "," + ",".join(["%.4f" % j for j in embed[i].tolist()])
            f.write(line + '\n')
