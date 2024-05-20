import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GATv2Conv
from args import AGATBD_args_parser
from torch_geometric.nn import global_mean_pool

args = AGATBD_args_parser()

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(args.device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
  if hard:
      shape = logits.size()
      _, k = y_soft.data.max(-1)
      y_hard = torch.zeros(*shape).to(args.device)
      y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
      y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
  else:
      y = y_soft
  return y


class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=args.heads, concat=False, dropout=args.dropout)  #in_feats是输入特征数，h_feats是输出特征数，heads是多头注意机制，negative_slope是leakyRELU的参数，默认0.2       
        self.conv2 = GATConv(h_feats, out_feats, heads=args.heads, concat=False, dropout=args.dropout)   #    concat如果是 False，多头注意机制就是平均而不是拼接
        # self.conv1 = GCNConv(in_feats, h_feats)  #in_feats是输入特征数，h_feats是输出特征数，heads是多头注意机制，negative_slope是leakyRELU的参数，默认0.2       
        # self.conv2 = GCNConv(h_feats, out_feats)   #    concat如果是 False，多头注意机制就是平均而不是拼接
    def forward(self, x, edge_index, edge_weight=None):
        # 24 128 / 2 118
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x

class AGATBD(nn.Module):
    def __init__(self, args):
        super(AGATBD, self).__init__()
        self.args = args
        self.num_nodes = args.input_size
        self.out_feats = args.out_feats
        self.embedding_dim = 100
        self.hidden_size = args.hidden_size
        self.dim_fc = (args.data_len - 18) * 16
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        
        self.gat = GAT(in_feats=args.seq_len, h_feats=args.h_feats, out_feats=self.out_feats)
        #self.gat = GAT_Encoder(input_dim=args.seq_len, hid_dim=32, gnn_embed_dim=128, dropout=0.5, heads=4)
        self.lstm = nn.GRU(input_size=args.input_size, hidden_size=self.hidden_size,
                            num_layers=args.num_layers, batch_first=True, dropout=args.dropout)
        #self.conv = torch.nn.Conv1d(1, 4, 10, stride=1)  # .to(device)
        self.fcs = nn.ModuleList()
        for k in range(args.input_size):    # 原模型的全连接层
            self.fcs.append(nn.Sequential(
                nn.Linear(self.out_feats + self.hidden_size, (self.out_feats + self.hidden_size) // 2),
                nn.ReLU(),
                nn.Linear((self.out_feats + self.hidden_size) // 2, args.output_size)
            ))
    
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot
        # Generate off-diagonal interaction graph
        off_diag = np.ones([4, 4])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(args.device)
        self.rel_send = torch.FloatTensor(rel_send).to(args.device)

    def forward(self, data, node_feas):
        # 最优图提取器
        # node_feas是原始训练数据，如果共n个电池，每个电池容量长度为l，则node_feas维度为(n, int(0.8*l))
        node_feas = node_feas.to(self.args.device)             
        x = node_feas.view(self.num_nodes, 1, -1)   
        x = self.conv1(x)                       
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)                       
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1)         
        x = self.fc(x)                         
        x = F.relu(x)
        x = self.bn3(x)
        
        receivers = torch.matmul(self.rel_rec, x)  
        senders = torch.matmul(self.rel_send, x)      
        x = torch.cat([senders, receivers], dim=1)    
        x = torch.relu(self.fc_out(x))                
        x = self.fc_cat(x)                            

        adj = gumbel_softmax(x, temperature=0.5, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(self.args.device)
        adj.masked_fill_(mask, 0)                 
        adj = sp.coo_matrix(adj.cpu().detach())  
        indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
        edge_index = torch.LongTensor(indices).to(self.args.device)  # PyG框架需要的coo形式
        
        # 空间、时间特征提取器和时空特征融合器
        # x是处理好的训练数据
        x, batch = data.x, data.batch      # x (batch_size*num_nodes , seq_len)

        batch_size = torch.max(batch).item() + 1
        x = self.gat(x, edge_index)   
       # print(x.shape)   # # x (batch_size*num_nodes , out_feas)
        batch_list = batch.cpu().numpy()
        # split
        xs = [[] for k in range(batch_size)]
        ys = [[] for k in range(batch_size)]
        for k in range(x.shape[0]):
            xs[batch_list[k]].append(x[k, :])
            ys[batch_list[k]].append(data.y[k, :])

        xs = [torch.stack(x, dim=0) for x in xs]
        ys = [torch.stack(x, dim=0) for x in ys]
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)
        # print(x.shape, y.shape)  #    (batchsize, 4, out_feas)  / (batchsize, 4, 1)
        gat_output = x
        x = x.permute(0, 2, 1)   #     (batchsize, out_feas, 4)
        x, _ = self.lstm(x)
        # print(x.shape)      #    (batchsize, out_feas, hiddensize)
        x = x[:, -1, :]
        # print(x.shape)      # 4, 128                    (batchsize, hiddensize)
        
        # 特征融合模块
        fuse = [[] for i in range(self.num_nodes)]
        gat_output = gat_output.permute(1, 0, 2)       # (4， batchsize, out_feas)

        for i in range(args.input_size):
            fuse[i] = torch.cat((gat_output[i], x), dim=1)
            fuse.append(torch.cat((gat_output[i][0], x[0]), dim=0))
        # print(fuse[0].shape)     # 一个列表，四个元素，每个元素是(batchsize, (out_feas + hidden_size)), 第N个元素是第N个变量和全部经过GRU的输出相拼接而成
        preds = []
        for i, fc in enumerate(self.fcs):    
            preds.append(fc(fuse[i]))
        pred = torch.stack(preds, dim=0)    
        pred = pred.permute(1, 0, 2)
        return pred, y