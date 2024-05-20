import os
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from models_new import AGATBD
def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))


def get_r2(y, pred):
    return r2_score(y, pred)


def get_mae(y, pred):
    return mean_absolute_error(y, pred)


def get_mse(y, pred):
    return mean_squared_error(y, pred)


def get_rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


@torch.no_grad()
def test(args, test_data, node_feas, pt_path):
    #graph_struct = []
    print('loading models...')
    model = AGATBD(args).to(args.device)
    print(pt_path)
    if os.path.exists(pt_path):
        print(11)
        model.load_state_dict(torch.load(pt_path)['model'])
    else:
        model.load_state_dict(torch.load('model_version_100.pkl')['model'])

    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.input_size)]
    preds = [[] for i in range(args.input_size)]
    for graph in tqdm(test_data):
        graph = graph.to(args.device)
        _pred, targets= model(graph, node_feas)     #
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)  

        for i in range(args.input_size):
            target = targets[:, i, :]    
            target = list(chain.from_iterable(target))
            ys[i].extend(target)
        for i in range(args.input_size):
            pred = _pred[:, i, :]
            pred = list(chain.from_iterable(pred.data.tolist()))
            preds[i].extend(pred)
    ys, preds = np.array(ys), np.array(preds)
    mses, rmses, maes, mapes = [], [], [], []
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print('--------------------------------')
        print('第', str(ind), '个变量:')
        print(len(y))
        mses.append(get_mse(y, pred))
        rmses.append(get_rmse(y, pred))
        maes.append(get_mae(y, pred))
        mapes.append(get_mape(y, pred))
        print('rmse:', get_rmse(y, pred))
        print('mae:', get_mae(y, pred))
    return ys, preds, mses, rmses, maes, mapes

# def plot(y, pred, ind, label):
#     # plot
#     fig = plt.figure()
#     plt.plot(pred, color='blue', label='pred value')
    
#     plt.plot(y, color='red', label='true value')
#     plt.title('第' + str(ind) + '变量的预测示意图')
#     plt.grid(True)
#     #plt.legend(loc='upper center', ncol=6)
#     #plt.show()
#     fig.savefig('/home/laicx/Codee/第{}个变量(model_version={}).jpg'.format(ind, args.model_version))


# args = gnn_lstm_args_parser()

# test(args=args, test_data=total_data, scaler=scaler)


