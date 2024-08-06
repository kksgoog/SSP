import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
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
        data_w = []
        data_s = []

        for file_path in data_classes:
            a = np.loadtxt(os.path.join(path, 'signal', file_path))

            n_w = wgn(a, -1)
            dataw = a + n_w

            n_s = wgn(a, -10)
            datas = a + n_s

            data_w.append(dataw)
            data_s.append(datas)
            data.append(np.loadtxt(
                os.path.join(path, 'signal', file_path)))

        data = np.array(data)
        data_w = np.array(data_w)
        data_s = np.array(data_s)
        data = np.pad(data, ((0, 0), (0, signal_size - data.shape[-1])),
                      'constant', constant_values=(0, 0))
        data_w = np.pad(data_w, ((0, 0), (0, signal_size - data_w.shape[-1])),
                      'constant', constant_values=(0, 0))
        data_s = np.pad(data_s, ((0, 0), (0, signal_size - data_s.shape[-1])),
                      'constant', constant_values=(0, 0))

        self.data = data[:, np.newaxis, :]
        self.data_w = data_w[:, np.newaxis, :]
        self.data_s = data_s[:, np.newaxis, :]

        # label
        label_classes = [d for d in os.listdir(os.path.join(path, 'label')) if
                         os.path.isfile(os.path.join(path, 'label', d))]
        label_classes.sort(key=lambda x: int(x[0:-4]))
        label = []
        for file_path in data_classes:
            label.append(np.loadtxt(
                os.path.join(path, 'label', file_path)))
        label = np.array(label)
        label -= 1
        self.label = label

    def get_data(self):
        return self.data, self.label



class GetDataset(Dataset):
    def __init__(self, data, label, is_ulb=False):
        self.data = data
        self.label = label
        self.is_ulb = is_ulb

    def wgn(self, data, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(data ** 2) / len(data)
        npower = xpower / snr
        return np.random.randn(len(data)) * np.sqrt(npower)


    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        a = torch.Tensor(data)
        a = np.array(a.view(-1))

        n_w = wgn(a, 1)
        data_w = a + n_w
        data_w = data_w.reshape(1, -1)

        n_s = wgn(a, -10)
        data_s = a + n_s
        data_s = data_s.reshape(1, -1)

        data = maxabs_scale(data, axis=1)
        data_w = maxabs_scale(data_w, axis=1)
        data_s = maxabs_scale(data_s, axis=1)

        if not self.is_ulb:
            return {'idx_lb': index, 'x_lb': torch.Tensor(data_w), 'y_lb': label}
        else:
            return {'idx_ulb': index, 'x_ulb_w': torch.Tensor(data_w), 'x_ulb_s': torch.Tensor(data_s)}
        # return torch.Tensor(data_w), torch.Tensor(data_s)

    def __len__(self):
        return len(self.data)
