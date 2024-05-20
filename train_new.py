import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import time
from models_new import AGATBD
import random

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(0)
def train(args, train_data, node_feas):
    
    model = AGATBD(args).to(args.device)
    model.train()
    loss_function_1 = nn.MSELoss().to(args.device)
    loss_function_2 = nn.L1Loss().to(args.device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    loss = []
    for epoch in tqdm(range(args.epochs), desc='Training'):
        train_loss = []
        epoch_loss = 0
        for graph in train_data:
            graph = graph.to(args.device)
            preds, labels = model(graph, node_feas)       
            total_loss = loss_function_1(preds, labels) + loss_function_2(preds, labels)
            epoch_loss = epoch_loss + total_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())
        loss.append(epoch_loss / len(train_data))
        scheduler.step()
        tqdm.write("epoch {:03d} train_loss {:.8f}".format(epoch, np.mean(train_loss)))

        
    state = {'model': model.state_dict()}
    name = f'{int(time.time())}' + '.pkl'
    path = 'models/' + name
    torch.save(state, path)
    return name



