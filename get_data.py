import random
import numpy as np
import os
import torch

import torch_geometric
from tqdm import tqdm
from args import AGATBD_args_parser
from torch_geometric.data import Data
# from torch_geometric.utils import to_undirected
#from get_data_CALCE import data_35, data_36, data_37, data_38


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def pre_process(data):
    # print(data)   # [4,168]
    node_feas = torch.tensor(np.array(data)).float()

    #print(data)   # data (4, 168)
    data = list(map(list, zip(*data)))    # 二维列表转置

    #print(np.array(data).shape)                           # data (168, 4)
    # print(type(data))                     # list   
    # print(np.array(data).shape)

    # print('划分数据集')   
    # train = data[:int(len(data) * 0.8)]
    # print(np.array(train).shape)            # (100, 4)
    # print(len(train))
    #print(train)
    #val = data[int(len(data) * 0.6):int(len(data) * 0.8)]  # 34 4
    # test = data[int(len(data) * 0.8):len(data)]
    # print(np.array(test).shape)            # (34, 4)

    # print(node_feas.T.shape)
    
    return node_feas, data

# 这个函数是用滑动窗口去处理原始数据，（4,168）会被处理成（168-seq_len）个（4，seq_len）个矩阵，标签是（168-seq_len）个(4,1),就是每个电池下一时刻的数据值
def process(args, dataset, batch_size, step_size, shuffle):

    print(len(dataset))
    # 这边不用理################################
    edge_index = [[], []]
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 3],
                            [1, 3, 0, 3, 3, 0, 2]])   
    edge_index = torch.LongTensor(edge_index)             
    # print(edge_index)    
    ######################################################

    seq = []
    graphs = []
    for i in tqdm(range(0, len(dataset) - args.seq_len - 1, step_size)):
        train_seq = []
        # print('get_data的seq_len:{}'.format(args.seq_len))
        # print('get_data的batchsize:{}'.format(batch_size))
        for j in range(i, i + args.seq_len):
            x = []
            for c in range(len(dataset[0])):  # 前8个时刻的所有变量
                x.append(dataset[j][c])
            train_seq.append(x)
        # 下1个时刻的所有变量
        train_labels = []
        for j in range(len(dataset[0])):
            train_label = []
            for k in range(i + args.seq_len, i + args.seq_len + 1):
                train_label.append(dataset[k][j])
            train_labels.append(train_label)
        # tensor
        train_seq = torch.FloatTensor(train_seq)
        train_labels = torch.FloatTensor(train_labels)
        #print(train_seq.shape, train_labels.shape)  # 8 4, 4 1

        # 此处可利用train_seq创建动态的邻接矩阵
        temp = Data(x=train_seq.T, y=train_labels, edge_index=edge_index)
        # print(temp)
        graphs.append(temp)
    train = graphs[:int(len(graphs) * 0.8)]
    val = graphs[int(len(graphs) * 0.6):int(len(graphs) * 0.8)]
    test = graphs[int(len(graphs) * 0.8):len(graphs)]
    train_data = torch_geometric.loader.DataLoader(train, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    val = torch_geometric.loader.DataLoader(val, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    test_data = torch_geometric.loader.DataLoader(test, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    total_data = torch_geometric.loader.DataLoader(graphs, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)
    return train_data, val, test_data, graphs, total_data
    # loader = torch_geometric.loader.DataLoader(graphs, batch_size=batch_size,
    #                                                 shuffle=shuffle, drop_last=False)
    # return loader

# train_data, Val, test_data, graphs, total_data = process(data, batch_size=args.batch_size, step_size=1, shuffle=False)


###########################################################################################################这边以下可以不用管，是拿来测试get_data预处理数据是否正确的

# if __name__ == '__main__':
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#     gat_mlp = GNN_LSTM(args).to(device)

#     j = 0
#     for i, data in enumerate(train_data):    # 共2795条训练数据,975条验证，975条测试
#         data = data.to(device)
#         # print(data)
#         # print(data.x)
#         # print(data.x.shape)
#         # print(data.y)
#         # #print(data[0].shape)         #(1, 24, 13)  (x (batch_size, input_size, seq_len) = (256, 13, 24))
#         # print(len(data))              #2
#         # print(data[0])
#         # print(data.batch)
#         # print(len(data.batch))
#         # print(type(data.batch))
#         pred, y , edge_index= gat_mlp(data, node_feas)

#         if i == 0:
#             break
#         j = i
#     print(j)

