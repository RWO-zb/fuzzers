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
    # Mock args object to store default configurations
    class Args:
        pass
    args = Args()


    
    # Keep original parameter configurations
    args.sparse = True
    args.layers = "500,300"
    args.layers = [int(l) for l in args.layers.split(",")]
    
    # Use small kernels for MountainCar
    args.kernels = "2,1,1"
    args.kernels = [int(l) for l in args.kernels.split(",")]
    args.filters = "256,256,128"
    args.filters = [int(l) for l in args.filters.split(",")]
    args.rp_params = '-1,3'
    args.rp_params = [float(l) for l in args.rp_params.split(",")]

    # Set RP parameters based on feature dimension
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

    # Initialize TapNet with parameters consistent with training (nfeat=Dimension, len_ts=Step)
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
    # Ensure correct shape
    if len(seq) < Hyperparameter.Step:
        # Pad sequence if too short
        padding = [[0] * Hyperparameter.Dimension] * (Hyperparameter.Step - len(seq))
        seq = seq + padding
    elif len(seq) > Hyperparameter.Step:
        # Truncate sequence if too long
        seq = seq[:Hyperparameter.Step]
    
    # seq is [Step, Dim] list
    
    # Convert to Tensor: [1, Step, Dim]
    siameseP2 = [seq]
    siameseP2 = torch.FloatTensor(np.array(siameseP2))
    
    if torch.cuda.is_available():
        siameseP2 = siameseP2.cuda()
    
    # Transpose input to (N, Dim, Step) format expected by the model
    if siameseP2.shape[2] == Hyperparameter.Dimension: # If the last dimension is the feature dimension
        siameseP2 = siameseP2.transpose(1, 2) # Change to (1, Dim, Step) format
    
    # Also transpose reference data if needed
    if bench_noCrash0.shape[2] == Hyperparameter.Dimension:
        bench_noCrash0 = bench_noCrash0.transpose(1, 2)

    # Inputs now match model expectations (1, Dim, Step)
    output1 = model(bench_noCrash0, siameseP2)
    output1 = torch.nn.Sigmoid()(output1)

    if output1[0][0] > 0.43:
        return 1
    else:
        return 0