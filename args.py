# -*- coding:utf-8 -*-
import argparse
import torch
import os

# torch.cuda.set_device(0)

# os.environ['CUDA_VISIBLE_DEVICES']='2'
def AGATBD_args_parser(data_len=168):

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=400, help='training epochs')
    parser.add_argument('--input_size', type=int, default=4, help='input dimension')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    # parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--device', default=torch.device("cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=50, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    
    parser.add_argument('--data_len', type=float, default=data_len, help='data len')
    
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')##############################
    parser.add_argument('--seq_len', type=int, default=8, help='seq len')#################################

    parser.add_argument('--heads', type=int, default=6, help='GATheads')
    
    parser.add_argument('--dropout', type=float, default=0.0, help='GATandGRU')
    parser.add_argument('--h_feats', type=int, default=16, help='GAT h_feats')
    parser.add_argument('--out_feats', type=int, default=32, help='GAT out_feats')
    parser.add_argument('--hidden_size', type=int, default=32, help='GRU hidden_size')
    parser.add_argument('--model_version', type=int, default=24, help='model_version')
    args = parser.parse_args()

    return args

def gnn_lstm_origin():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--input_size', type=int, default=13, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--lr', type=float, default=0.006, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=150, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    args = parser.parse_args()

    return args


