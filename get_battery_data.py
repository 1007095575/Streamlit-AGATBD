import random

import scipy.io
import numpy as np
from datetime import datetime
import scipy.interpolate as spi
import matplotlib.pyplot as plt

import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# convert str to datatime
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(
        hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


# get capacity data
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]

def getBatteryRct_Re(Battery):
    cycle, Rct, Re = [], [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'impedance':
            Rct.append(Bat['data']['Rct'][0])
            Re.append(Bat['data']['Re'][0])
            cycle.append(i)
            i += 1
    return [cycle, Rct], [cycle, Re]

def getBatteryImpedance(Battery):
    cycle, impedance, imp_temp = [], [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'impedance':
            for imp in range(len(Bat['data']['Rectified_Impedance'])) :
                imp_temp.append(Bat['data']['Rectified_Impedance'][imp])
            impedance_mean = np.mean(imp_temp)
            impedance.append(abs(impedance_mean))
            cycle.append(i)
            i += 1
    return [cycle, impedance]

# get the charge data of a battery
def getBatteryValues(Battery, Type='charge'):
    data = []
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data


def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split("/")[-1].split(".")[0]
    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        #if str(col[i][0][0]) != 'impedance':
        for j in range(len(k)):
            t = col[i][3][0][0][j][0];
            l = [t[m] for m in range(len(t))]
            d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(
            convert_to_time(col[i][2][0])), d2
        data.append(d1)
    return data


Battery_list = ['B0005', 'B0006', 'B0007']
dir_path = 'C:/Users/86180/Desktop/Codee/NASA/'


Battery = {}
for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    Battery[name + "_Capacity"] = getBatteryCapacity(data)
    Battery[name + "_Rct"], Battery[name+"_Re"] = getBatteryRct_Re(data)
    Battery[name + "_impedance"] = getBatteryImpedance(data)

#print(Battery['B0005_Capacity'])

x = []
for i in range(1, 169):
    x.append(i)

y = Battery['B0005_Capacity'][1]

xd = []
i = 1.5
while True:
    xd.append(i)
    i += 1
    if len(xd) == 110:
        break

xs = []
for j in range(0, 168):
    xs.append(x[j])
    if j < 110:
        xs.append(xd[j])
        
ipo = spi.splrep(x, y, k=3)
iy = spi.splev(xs, ipo)

#print(iy)
#print(len(iy))
# print('*' * 200)
# 归一化 [0, 1]
Battery['B0005_Capacity'][1] = iy
scaler = MinMaxScaler()
Battery['B0005_Capacity'][1] = scaler.fit_transform(np.array(Battery["B0005_Capacity"][1]).reshape(-1, 1)).reshape(-1,)
Battery['B0005_Rct'][1] = scaler.fit_transform(np.array(Battery["B0005_Rct"][1]).reshape(-1, 1)).reshape(-1,)
Battery['B0005_Re'][1] = scaler.fit_transform(np.array(Battery["B0005_Re"][1]).reshape(-1, 1)).reshape(-1,)
Battery['B0005_impedance'][1] = scaler.fit_transform(np.array(Battery["B0005_impedance"][1]).reshape(-1, 1)).reshape(-1,)

""" print(Battery["B0005_Capacity"][1])
print(len(Battery["B0005_Capacity"][1]))
print(Battery["B0005_Rct"][1])
print(len(Battery["B0005_Rct"][1]))
print(Battery["B0005_Re"][1])
print(len(Battery["B0005_Re"][1]))
print(Battery["B0005_impedance"][1])
print(len(Battery["B0005_impedance"][1])) """

# print("*" * 100)
# print(Battery["B0006_Capacity"])
# print(Battery["B0006_Rct"])
# print(Battery["B0006_Re"])
# print(Battery["B0006_impedance"])
#
# print("*" * 100)
# print(Battery["B0007_Capacity"])
# print(Battery["B0007_Rct"])
# print(Battery["B0007_Re"])
# print(Battery["B0007_impedance"])



def create_dataset(data: list, window_size: int):
    arr_x, arr_y = [], []
    for i in range(len(data) - window_size):
        x = data[i: i + window_size]
        y = data[i + window_size]
        arr_x.append(x)
        arr_y.append(y)
    return np.array(arr_x), np.array(arr_y)



window_size = 8
Battery_total_dataset = {}
Battery_train = {}
Battery_test = {}
Battery_labels = {}
labels_train = {}
labels_test = {}
Attribute = {'_Rct', '_Re', '_impedance'}

""" torch.manual_seed(0)
W1 = torch.empty(size=(278, 168))
#print(W1)
#nn.init.xavier_uniform_(W1.data, gain=1.414)
torch.nn.init.uniform_(W1.data, a=0, b=1) """

for name in Battery_list:
    print('handling ' + name + ' data...')
    Battery_total_dataset[name + '_Capacity'], Battery_labels[name] = create_dataset(Battery[name + '_Capacity'][1], window_size)
    for attribute in Attribute:
        Battery_total_dataset[name + attribute], _ = create_dataset(Battery[name + attribute][1], window_size)


print(Battery_total_dataset["B0005_Capacity"])
print(len(Battery_total_dataset["B0005_Capacity"]))
print(Battery_total_dataset["B0005_Rct"])
print(len(Battery_total_dataset["B0005_Rct"]))
print(Battery_total_dataset["B0005_Re"])
print(len(Battery_total_dataset["B0005_Re"]))
print(Battery_total_dataset["B0005_impedance"])
print(len(Battery_total_dataset["B0005_impedance"]))

split_ratio = 0.7

for name in Battery_list:
    len_train = int(len(Battery_total_dataset[name + '_Capacity']) * split_ratio)
    Battery_train[name + '_Capacity'] = Battery_total_dataset[name + '_Capacity'][:len_train]
    Battery_test[name + '_Capacity'] = Battery_total_dataset[name + '_Capacity'][len_train:]
    labels_train[name] = Battery_labels[name][:len_train]
    labels_test[name] = Battery_labels[name][len_train:]
    for attribute in Attribute:
        Battery_train[name + attribute] = Battery_total_dataset[name + attribute][:len_train]
        Battery_test[name + attribute] = Battery_total_dataset[name + attribute][len_train:]


# print(Battery_test['B0005_Capacity'])
# print(len(Battery_test['B0005_Capacity']))
# print(labels_test['B0005'])
# print(len(labels_test['B0005']))
# batch_size = 64
# ds = TensorDataset(X, Y)
# dl = DataLoader(ds, batch_size=batch_size, num_workers=0)
# ds_train = TensorDataset(X_train, Y_train)
# dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0)


# print(Battery_total_dataset.keys())
""" print(labels_train['B0005'])
print(type(labels_train['B0005']))
print(len(labels_train['B0005'])) """

print(Battery_train['B0005_Capacity'])
print(len(Battery_train['B0005_Capacity']))
print(type(Battery_train['B0005_Capacity'][0]))
print(Battery_test['B0005_Capacity'])
print(len(Battery_test['B0005_Capacity']))


# print(Battery_train['B0005_impedance'])
# print(len(Battery_train['B0005_impedance']))
# print(Battery_train['B0005_Rct'])
# print(len(Battery_train['B0005_Rct']))
# print(Battery_train['B0005_Re'])
# print(len(Battery_train['B0005_Re']))

# print('*' * 200)

# print(Battery_Train['B0006_Capacity'])
# print(len(Battery_Train['B0006_Capacity']))
# print(Battery_Train['B0006_impedance'])
# print(len(Battery_Train['B0006_impedance']))
# print(Battery_Train['B0006_Rct'])
# print(len(Battery_Train['B0006_Rct']))
# print(Battery_Train['B0006_Re'])
# print(len(Battery_Train['B0006_Re']))
# print(Battery['B0005_impedance'][1])
#
# print(type(Battery['B0005_impedance'][1]))
# print(torch.Tensor(Battery['B0005_impedance'][1]))
# print(torch.Tensor(Battery['B0005_impedance'][1]).shape)
# print(torch.Tensor(Battery['B0005_impedance'][1]).reshape(1,-1))
# print(torch.Tensor(Battery['B0005_impedance'][1]).reshape(1,-1).shape)
