# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from progress.bar import Bar

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score
from semilearn.core.utils import get_optimizer, get_cosine_schedule_with_warmup, get_logger, EMA
from torch.utils.tensorboard import SummaryWriter
from semi.core.criterions import CELoss
from operator import lt
import torch.nn.functional as F

def accuracy(output, target, topk=(1, )):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = lt((target - 6).view(1, -1).expand_as(pred), pred) & lt(pred,
                                                                          (target + 6).view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k / batch_size)
        return res

def get_acc(out, label):
    num_correct = 0
    total = out.shape[0]
    _, pred_label = out.max(1)
    for i in range(len(label)):
        if np.abs(pred_label[i].cpu().detach().numpy()-label[i].cpu().detach().numpy()) <= 6:
            num_correct += 1

    return num_correct / total


class Trainer:
    def __init__(self, config, algorithm, verbose=0):
        self.config = config
        self.verbose = verbose
        self.algorithm = algorithm
        self.ce_loss = CELoss()
        self.val_acc = []
        self.val_loss = []
        self.val_top1 = []
        self.start_epoch = 1
        self.train_sup = []
        self.train_unsup = []
        self.train_total = []
        self.train_top1 = []
        self.train_acc1 = []

        # TODO: support distributed training?
        torch.cuda.set_device(config.gpu)
        self.algorithm.model = self.algorithm.model.cuda(config.gpu)

        # setup logger
        self.save_path = os.path.join(config.save_dir, config.save_name)
        self.logger = get_logger(config.save_name, save_path=self.save_path, level="INFO")
        self.writer = SummaryWriter('logs')
        self.result_path = 'result2_ms_free_1e-4v0_saf'

    def plot_train_loss(self, step):
        path = os.path.join(os.path.join(self.result_path, str(step)), 'train_loss')
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

        # np.savetxt(os.path.join(path, 'train_loss.txt'), self.train_loss)
        np.savetxt(os.path.join(path, 'train_sup.txt'), self.train_sup)
        # np.savetxt(os.path.join(path, 'train_unsup.txt'), self.train_unsup)
        np.savetxt(os.path.join(path, 'train_total.txt'), self.train_total)

        # plt.plot(self.train_loss, label='train_loss')
        plt.plot(self.train_sup, label='sup_loss')
        plt.legend()
        plt.savefig(os.path.join(path, 'sup_loss.png'), dpi=600)
        plt.cla()

        # plt.plot(self.train_unsup, label='unsup_loss')
        # plt.legend()
        # plt.savefig(os.path.join(path, 'unsup_loss.png'), dpi=600)
        # plt.cla()

        plt.plot(self.train_total, label='total_loss')
        plt.legend()
        plt.savefig(os.path.join(path, 'total_loss.png'), dpi=600)
        plt.cla()

    def fit(self, train_lb_loader, train_ulb_loader, eval_loader):


        self.algorithm.loader_dict = {
            'train_lb': train_lb_loader,
            'train_ulb': train_ulb_loader,
            'eval': eval_loader
        }
        self.algorithm.model.train()
        # train
        self.algorithm.it = 0
        self.algorithm.best_eval_acc = 0.0
        self.algorithm.best_epoch = 0
        self.algorithm.call_hook("before_run")

        for epoch in range(self.start_epoch, self.config.epoch+1):
            self.algorithm.epoch = epoch
            sup = 0
            unsup = 0
            total = 0
            y_true = []
            y_pred = []
            y_logits = []
            print("Epoch: {}".format(epoch))
            if self.algorithm.it > self.config.num_train_iter:
                break

            bar = Bar('Processing', max=len(train_lb_loader))

            self.algorithm.model.train()
            self.algorithm.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(train_lb_loader, train_ulb_loader):

                if self.algorithm.it > self.config.num_train_iter:
                    break

                self.algorithm.call_hook("before_train_step")
                out_dict, log_dict, sup_loss, unsup_loss, total_loss, pseudo_label, logits_x_ulb_s = self.algorithm.train_step(**self.algorithm.process_batch(**data_lb, **data_ulb))
                sup += sup_loss
                unsup += unsup_loss
                total += total_loss
                self.algorithm.out_dict = out_dict
                self.algorithm.log_dict = log_dict

                y_true.extend(pseudo_label.cpu().tolist())
                y_pred.extend(torch.max(logits_x_ulb_s, dim=-1)[1].cpu().tolist())
                y_logits.append(torch.softmax(logits_x_ulb_s, dim=-1).cpu().detach().numpy())
                self.algorithm.call_hook("after_train_step")

                bar.suffix = ("Iter: {batch:4}/{iter:4}.".format(batch=self.algorithm.it, iter=len(train_lb_loader)))
                bar.next()
                self.algorithm.it += 1
            bar.finish()

            self.train_sup.append(float(sup/self.config.num_eval_iter))
            self.train_unsup.append(float(unsup/self.config.num_eval_iter))
            self.train_total.append(float(total/self.config.num_eval_iter))

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_logits = np.concatenate(y_logits)
            top1 = accuracy_score(y_true, y_pred)

            acc1, acc5 = accuracy(torch.tensor(y_logits), torch.tensor(y_true), topk=(1, 5))
            acc1 = float(np.array(acc1))
            self.train_acc1.append(float(acc1))
            self.train_top1.append(float(top1))

            if epoch % 10 == 0:
                self.plot_train_loss(epoch)

            self.algorithm.call_hook("after_train_epoch")

            # validate
            result = self.evaluate(eval_loader, epoch=epoch)

            # save model
            self.algorithm.save_model('latest_model.pth', self.save_path)

            # best
            if result['acc'] > self.algorithm.best_eval_acc:
                self.algorithm.best_eval_acc = result['acc']
                self.algorithm.best_epoch = self.algorithm.epoch
                self.algorithm.save_model('model_best.pth', self.save_path)

        self.logger.info(
            "Best acc {:.4f} at epoch {:d}".format(self.algorithm.best_eval_acc, self.algorithm.best_epoch))
        self.logger.info("Training finished.")

    def plot_loss_and_acc(self, step):
        path = os.path.join(os.path.join(self.result_path, str(step)), 'loss&acc')
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

        # np.savetxt(os.path.join(path, 'train_loss.txt'), self.train_loss)
        np.savetxt(os.path.join(path, 'val_loss.txt'), self.val_loss)

        # plt.plot(self.train_loss, label='train_loss')
        plt.plot(self.val_loss, label='val_loss')
        plt.legend()
        plt.savefig(os.path.join(path, 'loss.png'), dpi=600)
        plt.cla()

        # np.savetxt(os.path.join(self.result_path, "pred.txt"), np.array(pred_val))
        # np.savetxt(os.path.join(path, 'train_acc.txt'), self.train_acc)
        np.savetxt(os.path.join(path, 'val_acc.txt'), self.val_acc)
        # plt.plot(self.train_acc, label='train_acc')
        plt.plot(self.val_acc, label='val_acc')
        plt.legend()
        plt.savefig(os.path.join(path, 'acc.png'), dpi=600)
        plt.cla()

    def plot_top1(self, step):
        path = os.path.join(os.path.join(self.result_path, str(step)), 'acc')
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

        # np.savetxt(os.path.join(path, 'train_loss.txt'), self.train_loss)
        np.savetxt(os.path.join(path, 'val_top1.txt'), self.val_top1)

        # plt.plot(self.train_loss, label='train_loss')
        plt.plot(self.val_top1, label='val_top1')
        plt.legend()
        plt.savefig(os.path.join(path, 'top1.png'), dpi=600)
        plt.cla()


    def evaluate(self, data_loader, epoch, use_ema_model=False):

        y_pred, y_logits, y_true, val_acc, val_loss = self.predict(data_loader, use_ema_model, return_gt=True)


        top1 = accuracy_score(y_true, y_pred)
        self.val_top1.append(float(top1))

        val_acc = np.array(val_acc/len(data_loader))
        val_loss = np.array(val_loss/len(data_loader))

        self.val_loss.append(float(val_loss))
        self.val_acc.append(float(val_acc))

        if epoch % 10 == 0:
            self.plot_loss_and_acc(epoch)
            self.plot_top1(epoch)

        acc1, acc5 = accuracy(torch.tensor(y_logits), torch.tensor(y_true), topk=(1, 5))
        acc1 = float(np.array(acc1))
        acc5 = float(np.array(acc5))


        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')


        result_dict = {'epoch': epoch, 'acc': top1, 'precision': precision, 'recall': recall, 'f1': f1, 'acc_ms': val_acc,
                       'acc1': acc1, 'acc5': acc5}
        # print(self.val_acc)
        self.logger.info("evaluation metric")
        for key, item in result_dict.items():
            self.logger.info("{:s}: {:.4f}".format(key, item))
        self.writer.close()
        return result_dict


    def predict(self, data_loader, use_ema_model=False, return_gt=False):
        self.algorithm.model.eval()
        if use_ema_model:
            self.algorithm.ema.apply_shadow()

        y_true = []
        y_pred = []
        y_logits = []
        total_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        total_num = 0.0

        with torch.no_grad():
            for data in data_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.config.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.config.gpu)
                y = y.cuda(self.config.gpu)

                logits = self.algorithm.model(x)['logits']
                y = y.to(dtype=torch.long)
                out = logits.to(device=self.config.gpu)
                loss = F.cross_entropy(out, y)

                val_acc += get_acc(out, y)
                val_loss += loss.item()




                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())



        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)



        if use_ema_model:
            self.algorithm.ema.restore()
        self.algorithm.model.train()
        
        if return_gt:
            return y_pred, y_logits, y_true, val_acc, val_loss
        else:
            return y_pred, y_logits, val_acc, val_loss