import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.preprocessing import maxabs_scale
import os


def wgn(data, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(data ** 2) / len(data)
    npower = xpower / snr
    return np.random.randn(len(data)) * np.sqrt(npower)



class GetData:
    def __init__(self, path, signal_size=4096):
        # data
        data_classes = [d for d in os.listdir(os.path.join(path, 'signal')) if
                        os.path.isfile(os.path.join(path, 'signal', d))]
        data_classes.sort(key=lambda x: int(x[0:-4]))
        data = []
        for file_path in data_classes:
            data.append(np.loadtxt(
                os.path.join(path, 'signal', file_path)))

        data = np.array(data)
        data = np.pad(data, ((0, 0), (0, signal_size - data.shape[-1])),
                      'constant', constant_values=(0, 0))
        # data = maxabs_scale(data, axis=0)
        self.data = data[:, np.newaxis, :]

    def get_data(self):
        return self.data



class GetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data = self.data[index]

        # a = torch.Tensor(data)
        # a = np.array(a.view(-1))

        # n_s = wgn(a, -1)
        # data_s = a + n_s
        # data_s = data_s.reshape(1, -1)
        # data_s = maxabs_scale(data_s, axis=1)

        file1 = '/data0/zzr_data/realnoisy/13.txt'
        # file1 = 'D:/realnoisy/13.txt'
        a = np.loadtxt(file1)
        # a = maxabs_scale(a, axis=0)
        a = a*0.06
        a = np.pad(a, (0, 4096 - a.shape[-1]),
                   'constant', constant_values=0)
        data_s = data+a
        # data_s = maxabs_scale(data_s, axis=0)
        return torch.Tensor(data_s)

    def __len__(self):
        return len(self.data)
