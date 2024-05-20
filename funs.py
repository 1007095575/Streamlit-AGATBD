from get_data import pre_process, process
from test_new import test
from train_new import train
from args import AGATBD_args_parser
import os



def predicting(args, data, pt_name=''):
    # args = AGATBD_args_parser()
    path = 'models/' + pt_name
    node_feas, data = pre_process(data)
    _, _, _, _, total_data = process(args, data, batch_size=args.batch_size, step_size=1, shuffle=False)
    return test(args=args, test_data=total_data, node_feas=node_feas, pt_path=path) # ys, preds, mses, rmses, maes, mapes


def training(args, data, params):
    # args = AGATBD_args_parser()
    for k, v in params.items():
        setattr(args, k, v)
    print(args)
    node_feas, data = pre_process(data)
    train_data, _, _, _, _ = process(args, data, batch_size=args.batch_size, step_size=1, shuffle=False)
    return train(args=args, train_data=train_data, node_feas=node_feas) # path

def get_model_data(pt_name):
    if pt_name == '默认' or pt_name == '':
        return None
    path = 'models/' + pt_name
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data  = f.read()
        return data
    return None

def get_model_list():
    model_list = os.listdir('models')
    ls = ['默认'] + model_list
    #print(ls)   # ['默认', '1710646323.pkl', '1710646414.pkl', '1710646496.pkl', '1710647185.pkl', '1710647852.pkl', '1710647955.pkl', '1710648040.pkl']
    return ls