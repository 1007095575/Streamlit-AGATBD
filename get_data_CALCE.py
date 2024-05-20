import scipy.io
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
import os
import scipy.io
import numpy as np

import scipy.interpolate as spi
import matplotlib.pyplot as plt
from pandas import Series
import torch
from models_new import AGATBD

import torch_geometric
from tqdm import tqdm
from args import AGATBD_args_parser
from torch_geometric.data import Data

from sklearn.preprocessing import MinMaxScaler

args = AGATBD_args_parser()

# data = h5py.File('/home/laicx/GSL_GAT_LSTM_ing_two/CALCE/CX2.mat')
#matfile = 'C:/Users/86180/Desktop/Cite related paper(s) if these files help you/Shared Data (public data)/Public data/CS2.mat'
#data = scipy.io.loadmat(matfile)
# print(data.keys())
# print(data.values())
# print(data['CS2_35'].shape)

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# data_34 = np.transpose(data['CX2_34'])[0][:1243]
# data_36 = np.transpose(data['CX2_36'])[0][:1243]
# data_37 = np.transpose(data['CX2_37'])[0][:]
# data_38 = np.transpose(data['CX2_38'])[0][:1243]

# # print(len(data_34))    # 长度为1243
# # print(len(data_36))
# # print(len(data_37))
# # print(len(data_38))

# # data (4, 278)
# data_noScaler = [[] for i in range(args.input_size)]
# data_noScaler[0] = data_34
# data_noScaler[1] = data_36
# data_noScaler[2] = data_37
# data_noScaler[3] = data_38
# data = [[] for i in range(args.input_size)]
# scaler = MinMaxScaler()

# data[0] = scaler.fit_transform(np.array(data_34).reshape(-1, 1)).reshape(-1,)
# data[1] = scaler.fit_transform(np.array(data_36).reshape(-1, 1)).reshape(-1,)
# data[2] = scaler.fit_transform(np.array(data_37).reshape(-1, 1)).reshape(-1,)
# data[3] = scaler.fit_transform(np.array(data_38).reshape(-1, 1)).reshape(-1,)

# # print(len(data[0]))  # 1243


# # data[1] = scaler.fit_transform(data_35)
# # data[2] = scaler.fit_transform(data_35)
# # data[3] = scaler.fit_transform(data_35)

# node_feas = torch.tensor(np.array(data)).float()    #(4, 1243)

# #print(data)   # data (4, 168)
# data = list(map(list, zip(*data)))    # 二维列表转置
# # #print(data)                           # data (1243, 4)
# # print(type(data))                     # list   
# # print(np.array(data).shape)

# # print('划分数据集')
# # train = data[:int(len(data) * 0.8)]
# # print(np.array(train).shape)            # (100, 4)
# # print(len(train))
# # #print(train)
# # #val = data[int(len(data) * 0.6):int(len(data) * 0.8)]  # 34 4
# # test = data[int(len(data) * 0.8):len(data)]
# # print(np.array(test).shape)            # (34, 4)

# # print(node_feas.T.shape)
# # def calc_corr(a, b):
# #     s1 = Series(a)
# #     s2 = Series(b)
# #     return s1.corr(s2)

# # edge_index = [[], []]
# # # 计算相关系数
# # # data (x, num_nodes)
# # for i in range(args.input_size):
# #     for j in range(i + 1, args.input_size):
# #         x, y = np.array(data)[:, i], np.array(data)[:, j]
# #         corr = calc_corr(x, y)
# #         if corr >= 0.6:
# #             edge_index[0].append(i)
# #             edge_index[1].append(j)

# # 全连接
# # edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
# #                            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]])    #47 

# # model_version = 1  
# # edge_index = torch.tensor([[0, 3, 3, 3],
# #                            [2, 0, 1, 2]])    # 2

# # model_version = 3 
# # edge_index = torch.tensor([[0, 0, 0, 1, 2, 2, 3],
# #                            [1, 2, 3, 3, 0, 1, 0]])  #4
# # edge_index = torch.tensor([[1, 1, 2, 2, 3, 3],
# #                            [0, 2, 1, 3, 0, 1]])   # 5
# # edge_index = torch.tensor([[0, 1, 1, 2, 3],
# #                            [3, 0, 3, 0, 2]])    #6
# # edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 3, 3],
# #                            [1, 2, 3, 0, 2, 1, 0, 2]])    #7
# # edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 3],
# #                             [1, 2, 3, 0, 2, 0, 0]])    # 8
# # edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3],
# #                            [1, 2, 3, 0, 2, 3, 1, 3, 0, 1, 2]])   # 9

# # model_version = 10
# # edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 3, 3],
# #                            [1, 2, 3, 0, 2, 3, 3, 0, 2]])    # 11
# # edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 3, 3, 3],
# #                            [1, 2, 3, 2, 3, 0, 0, 1, 2]])   # 12
# # edge_index = torch.tensor([[0, 2, 2, 3],
# #                            [2, 0, 3, 1]])  #13
# # edge_index = torch.tensor([[0, 1, 2, 2, 3, 3],
# #                            [3, 2, 1, 3, 0, 2]])    #14 

# # model_version = 15
# # edge_index = torch.tensor([[0, 2, 2, 3, 3],
# #                            [2, 0, 3, 0, 2]])    #16 
# # edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 3],
# #                            [1, 2, 3, 0, 2, 0, 1, 3, 0]])   # 17
# edge_index = torch.tensor([[0, 0, 1, 2, 2, 2, 3],
#                            [1, 2, 0, 0, 1, 3, 0]])  #18
# # edge_index = torch.tensor([[0, 0, 1, 1, 3, 3, 3],
#                         #    [1, 3, 0, 3, 0, 1, 2]])    #19

# # model_version = 20
# edge_index = torch.tensor([[0, 0, 1, 3, 3],
#                            [2, 3, 3, 0, 2]])    # 22
# # model_version = 21
# edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3],
#                            [1, 2, 3, 0, 2, 3, 0, 3, 0]])   # 24
# # edge_index = torch.tensor([[0, 2, 2, 3],
# #                            [2, 0, 3, 1]])  #13
# # edge_index = torch.tensor([[0, 1, 2, 2, 3, 3],
# #                            [3, 2, 1, 3, 0, 2]])    #14 
# # edge_index = torch.tensor([[0, 2, 2, 3, 3],
# #         [2, 0, 3, 0, 2]])    #16 


# edge_index = torch.LongTensor(edge_index)                #      edge_index的来源有问题
# print(edge_index)     #tensor([[1, 1, 2],[2, 3, 3]])
# # edge_index = to_undirected(edge_index, num_nodes=args.input_size)
# # print(edge_index)

# def process(dataset, batch_size, step_size, shuffle):
#     seq = []
#     graphs = []
#     for i in tqdm(range(0, len(dataset) - args.seq_len - 1, step_size)):
#         train_seq = []
#         for j in range(i, i + args.seq_len):
#             x = []
#             for c in range(len(dataset[0])):  # 前8个时刻的所有变量
#                 x.append(dataset[j][c])
#             train_seq.append(x)
#         # 下1个时刻的所有变量
#         train_labels = []
#         for j in range(len(dataset[0])):
#             train_label = []
#             for k in range(i + args.seq_len, i + args.seq_len + 1):
#                 train_label.append(dataset[k][j])
#             train_labels.append(train_label)
#         # tensor
#         train_seq = torch.FloatTensor(train_seq)
#         train_labels = torch.FloatTensor(train_labels)
#         #print(train_seq.shape, train_labels.shape)  # 8 4, 4 1

#         # 此处可利用train_seq创建动态的邻接矩阵
#         temp = Data(x=train_seq.T, y=train_labels, edge_index=edge_index)
#         # print(temp)
#         graphs.append(temp)
#     train = graphs[:int(len(graphs) * 0.8)]
#     val = graphs[int(len(graphs) * 0.6):int(len(graphs) * 0.8)]
#     test = graphs[int(len(graphs) * 0.8):len(graphs)]
#     train_data = torch_geometric.loader.DataLoader(train, batch_size=batch_size,
#                                                    shuffle=shuffle, drop_last=False)
#     val = torch_geometric.loader.DataLoader(val, batch_size=batch_size,
#                                                    shuffle=shuffle, drop_last=False)
#     test_data = torch_geometric.loader.DataLoader(test, batch_size=batch_size,
#                                                    shuffle=shuffle, drop_last=False)
#     total_data = torch_geometric.loader.DataLoader(graphs, batch_size=batch_size,
#                                                    shuffle=shuffle, drop_last=False)
#     return train_data, val, test_data, graphs, total_data
#     # loader = torch_geometric.loader.DataLoader(graphs, batch_size=batch_size,
#     #                                                 shuffle=shuffle, drop_last=False)
#     # return loader
# train_data, Val, test_data, graphs, total_data = process(data, batch_size=args.batch_size, step_size=1, shuffle=False)


# # if __name__ == '__main__':
# #     # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# #     device = torch.device("cpu")
# #     gat_mlp = GNN_LSTM(args).to(device)

# #     j = 0
# #     for i, data in enumerate(train_data):    # 
# #         data = data.to(device)
# #         # print(data)
# #         # print(data.x)
# #         # print(data.y)
# #         # #print(data[0].shape)         #(1, 24, 13)  (x (batch_size, input_size, seq_len) = (256, 13, 24))
# #         # print(len(data))              #2
# #         # print(data[0])
# #         # print(data.batch)
# #         # print(len(data.batch))
# #         # print(type(data.batch))
# #         pred, y, edge_index= gat_mlp(data, node_feas)

# #         if i == 0:
# #             break
# #         j = i
# #     print(j)