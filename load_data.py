import pandas as pd
import numpy as np
import torch
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset
from logging import getLogger


logger = getLogger(__name__)


def load_data():
    logger.debug('enter')
    data = load_airline()
    data_scaled = minmax_scale(data)
    train_data, test_data = temporal_train_test_split(data_scaled)
    logger.debug('exit')
    return train_data, test_data


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.teacher = y

    def __len__(self):
        return len(self.teacher)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.teacher[idx]

        return out_data, out_label


def get_data(timestep=6):
    train_data, test_data = load_data()
    X_train, y_train, X_test, y_test = [], [], [], []

    for n in range(len(train_data)-6-timestep):
        X_train.append(train_data[n:n+timestep])
        y_train.append(train_data[n+timestep+6])

    for n in range(len(test_data)-6-timestep):
        X_test.append(test_data[n:n+timestep])
        y_test.append(test_data[n+timestep+6])

    trainSet = MyDataset(torch.FloatTensor(np.array(X_train)).cuda(), torch.FloatTensor(np.array(y_train)).cuda())
    testSet = MyDataset(torch.FloatTensor(np.array(X_test)).cuda(), torch.FloatTensor(np.array(y_test)).cuda())
    return trainSet, testSet


def generate_privilege_information(timestep=6):
    train_data, test_data = load_data()
    X_train, y_train, X_test, y_test = [], [], [], []

    for n in range(len(train_data)-6-timestep):
        X_train.append(train_data[n:n+timestep+6])
        y_train.append(train_data[n+timestep+6])

    for n in range(len(test_data)-6-timestep):
        X_test.append(test_data[n:n+timestep+6])
        y_test.append(test_data[n+timestep+6])

    trainSet = MyDataset(torch.FloatTensor(np.array(X_train)).cuda(), torch.FloatTensor(np.array(y_train)).cuda())
    testSet = MyDataset(torch.FloatTensor(np.array(X_test)).cuda(), torch.FloatTensor(np.array(y_test)).cuda())
    return trainSet, testSet



if __name__ == '__main__':
    train_df, test_df = load_data()
    print(train_df.shape)
    print(test_df.shape)

    trainset, testset = get_data()
    print(len(trainset))
    print(len(testset))
    tmp_data, tmp_label = trainset[0]
    print(len(tmp_data))
    print(tmp_label)
    tmp_data, tmp_label = testset[0]
    print(len(tmp_data))
    print(tmp_label)

    privilege_trainset, privilege_testset = generate_privilege_information()
    tmp_data, tmp_label = privilege_trainset[0]
    print(len(tmp_data))
    print(tmp_label)
    tmp_data, tmp_label = privilege_testset[0]
    print(len(tmp_data))
    print(tmp_label)
