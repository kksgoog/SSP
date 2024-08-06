import os
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch._C import device
from glob import glob
from data import *
from net_ms import *
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


class Train:
    def __init__(self,
                 train_path='data/train',
                 test_path='data/test',
                 result_path='result',
                 signal_size=4096,
                 num_epochs=200,
                 batch_size=50,
                 input_nc=1,
                 output_nc=1,
                 sub_nums=4,
                 alpha=1.0,
                 milestones=[10, 20, 30],
                 lr=1e-4,
                 weight_decay=1e-4,
                 decay_flag=True,
                 device='cuda:0',
                 resume=False):
        self.train_path = train_path
        self.test_path = test_path
        self.result_path = result_path
        self.signal_size = signal_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.sub_nums = sub_nums
        self.alpha = alpha
        self.milestones = milestones
        self.lr = lr
        self.weight_decay = weight_decay
        self.decay_flag = decay_flag
        self.device = device
        self.resume = resume

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.pred_val = []
        self.true_val = []
        self.start_epoch = 1

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def dataload(self):
        getdata = GetData(self.train_path)
        data, label = getdata.get_data()

        train_size = int(len(data) * 0.85)
        val_size = len(data) - train_size
        train_iter, val_iter = torch.utils.data.random_split(GetDataset(data, label), [train_size, val_size])

        self.train_iter = DataLoader(train_iter, batch_size=self.batch_size, shuffle=True)
        self.val_iter = DataLoader(val_iter, batch_size=self.batch_size, shuffle=True)

        getdata = GetData(self.test_path)
        test_data, test_label = getdata.get_data()
        self.test_iter = DataLoader(GetDataset(test_data, test_label), batch_size=1, shuffle=True)

    def build_model(self):
        self.net = UNet().to(device=self.device)

    def loss(self):
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.001).to(device=self.device)

    def define_optim(self):
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = MultiStepLR(self.optim, milestones=[int(x) for x in self.milestones])

    def save_model(self, path, step):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        params = {}
        params['net'] = self.net.state_dict()
        params['optim'] = self.optim.state_dict()
        params['scheduler'] = self.scheduler.state_dict
        params['train_loss'] = self.train_loss
        params['val_loss'] = self.val_loss
        params['train_acc'] = self.train_acc
        params['val_acc'] = self.val_acc
        params['start_epoch'] = self.start_epoch
        torch.save(params, os.path.join(path, 'model_params_%07d.pt' % step))

    def load_model(self, path, step):
        params = torch.load(os.path.join(path, 'model_params_%07d.pt' % step))
        self.net.load_state_dict(params['net'])
        self.optim.load_state_dict(params['optim'])
        # self.scheduler.load_state_dict(params['scheduler'])
        self.train_loss = params['train_loss']
        self.val_loss = params['val_loss']
        self.train_acc = params['train_acc']
        self.val_acc = params['val_acc']
        self.start_epoch = params['start_epoch']

    def plot_loss_and_acc(self, step):
        path = os.path.join(os.path.join(self.result_path, str(step)), 'loss&acc')
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        np.savetxt(os.path.join(path, 'train_loss.txt'), self.train_loss)
        np.savetxt(os.path.join(path, 'val_loss.txt'), self.val_loss)

        plt.plot(self.train_loss, label='train_loss')
        plt.plot(self.val_loss, label='val_loss')
        plt.legend()
        plt.savefig(os.path.join(path, 'loss.png'), dpi=600)
        plt.cla()

        # np.savetxt(os.path.join(self.result_path, "pred.txt"), np.array(pred_val))
        np.savetxt(os.path.join(path, 'train_acc.txt'), self.train_acc)
        np.savetxt(os.path.join(path, 'val_acc.txt'), self.val_acc)
        plt.plot(self.train_acc, label='train_acc')
        plt.plot(self.val_acc, label='val_acc')
        plt.legend()
        plt.savefig(os.path.join(path, 'acc.png'), dpi=600)
        plt.cla()


        # np.savetxt(os.path.join(path, "pred.txt"), self.pred_val, fmt='%d')
        # np.savetxt(os.path.join(path, "true.txt"), self.true_val, fmt='%d')

    def plot_result(self, result, step, num):
        path = os.path.join(os.path.join(
            self.result_path, str(step)), 'result')
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        np.savetxt(os.path.join(path, 'result{}.txt'.format(num)), result)
        plt.plot(result, label='result')
        plt.legend()
        plt.savefig(os.path.join(path, 'result{}.png'.format(num)), dpi=600)
        plt.cla()

    def plot_feature(self, feature, step, num):
        path = os.path.join(os.path.join(
            self.result_path, str(step)), 'last_feature')
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        plt.plot(feature, label='last_feature')
        # print(feature)
        plt.legend()
        plt.savefig(os.path.join(path, 'last_feature{}.png'.format(num)), dpi=600)
        plt.cla()

    def plot_feature0(self, feature, step, num):
        path = os.path.join(os.path.join(
            self.result_path, str(step)), 'stem_feature')
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        plt.plot(feature, label='stem_feature')
        # print(feature)
        plt.legend()
        plt.savefig(os.path.join(path, 'stem_feature{}.png'.format(num)), dpi=600)
        plt.cla()

    def plot_feature1(self, feature, step, num):
        path = os.path.join(os.path.join(
            self.result_path, str(step)), 'down1_feature')
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        plt.plot(feature, label='down1_feature')
        # print(feature)
        plt.legend()
        plt.savefig(os.path.join(path, 'down1_feature{}.png'.format(num)), dpi=600)
        plt.cla()

    def plot_feature2(self, feature, step, num):
        path = os.path.join(os.path.join(
            self.result_path, str(step)), 'down2_feature')
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        plt.plot(feature, label='down2_feature')
        # print(feature)
        plt.legend()
        plt.savefig(os.path.join(path, 'down2_feature{}.png'.format(num)), dpi=600)
        plt.cla()



    @staticmethod
    def get_acc(out, label):
        num_correct = 0
        total = out.shape[0]
        _, pred_label = out.max(1)
        for i in range(len(label)):
            if np.abs(pred_label[i].cpu().detach().numpy()-label[i].cpu().detach().numpy()) <= 5:
                num_correct += 1

        return num_correct / total

    def train(self):
        if self.resume:
            model_list = glob(os.path.join(self.result_path, 'model', '*.pt'))
            if len(model_list) != 0:
                model_list.sort()
                start_step = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load_model(os.path.join(self.result_path, 'model'), start_step)
            print("load success")

        for epoch in range(self.start_epoch, 1 + self.num_epochs):
            self.net.train()
            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0

            loop = tqdm(self.train_iter, desc='Train')

            for x, y in loop:
                self.optim.zero_grad()
                x, y = x.to(dtype=torch.float, device=self.device), y.to(dtype=torch.long, device=self.device)
                out = self.net(x)
                out = out.view(out.shape[0], -1)
                out = out.to(dtype=torch.float, device=self.device)

                loss = self.loss(out, y)
                loss.backward()
                self.optim.step()
                train_loss += loss.item()
                train_acc += self.get_acc(out, y)

            self.scheduler.step()
            train_loss = train_loss/len(self.train_iter)
            train_acc = train_acc/len(self.train_iter)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)


            #val
            self.net.eval()

            for x, y in self.val_iter:
                x, y = x.to(dtype=torch.float, device=self.device), y.to(dtype=torch.long, device=self.device)
                out = self.net(x)
                out = out.view(out.shape[0], -1)
                out = out.to(dtype=torch.float, device=self.device)
                loss = self.loss(out, y)
                val_loss += loss.item()
                val_acc += self.get_acc(out, y)
                _, pred_label = out.max(1)
                self.pred_val.append(pred_label.cpu().detach().numpy())
                self.true_val.append(y.cpu().detach().numpy())


            val_loss = val_loss/len(self.val_iter)
            val_acc = val_acc/len(self.val_iter)
            self.val_loss.append(float(val_loss))
            self.val_acc.append(float(val_acc))

            print("Epoch %d: Train Loss %f, Train Acc %f, Val Loss: %f, Val Acc: %f,"
                  % (epoch, train_loss, train_acc, val_loss, val_acc))

            if epoch % 10 == 0:
                self.plot_loss_and_acc(epoch)

            if epoch % 40 == 0:
                num = 0
                for x, y in self.test_iter:
                    x, y = x.to(dtype=torch.float, device=self.device), y.to(dtype=torch.long, device=self.device)
                    out = self.net(x)
                    out = torch.exp(out)
                    out = out.view(out.shape[0], -1)
                    out = out.to(dtype=torch.float, device='cpu').detach().numpy()

                    features = self.net.last_feature.view(out.shape[0], -1).cpu().detach().numpy()
                    feature0 = self.net.stem_feature.view(out.shape[0], -1).cpu().detach().numpy()
                    feature1 = self.net.down1_feature.view(out.shape[0], -1).cpu().detach().numpy()
                    feature2 = self.net.down2_feature.view(out.shape[0], -1).cpu().detach().numpy()

                    for i in range(out.shape[0]):
                        num += 1
                        result = out[i]
                        feature = features[i]
                        feature_0 = feature0[i]
                        feature_1 = feature1[i]
                        feature_2 = feature2[i]
                        self.plot_result(result, epoch, num)
                        self.plot_feature(feature, epoch, num)
                        self.plot_feature0(feature_0, epoch, num)
                        self.plot_feature1(feature_1, epoch, num)
                        self.plot_feature2(feature_2, epoch, num)


            if epoch % 50 == 0:
                self.start_epoch = epoch
                self.save_model(os.path.join(self.result_path, 'model'), epoch)
                # np.savetxt(os.path.join(self.result_path, "train_loss.txt"), np.array(self.train_loss))
                # np.savetxt(os.path.join(self.result_path, "valid_acc.txt"), np.array(self.valid_acc))


if __name__ == '__main__':
    train = Train(train_path='./data_5k',
                  test_path='D:/data/test',
                  result_path='result_net_ms',
                  signal_size=4096,
                  num_epochs=200,
                  batch_size=1,
                  sub_nums=4,
                  alpha=0.5,
                  milestones=[10, 20, 30],
                  lr=1e-4,
                  weight_decay=0.01,
                  decay_flag=True,
                  device='cuda:0',
                  resume=False)

    train.setup_seed(2024)
    train.dataload()
    train.build_model()
    train.loss()
    train.define_optim()
    train.train()
    print("training finished!")























