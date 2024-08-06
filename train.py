import os
import numpy as np
from torchvision import transforms

from semi import get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from semi import split_ssl_data
from sklearn.model_selection import train_test_split
from dataset import *
from matplotlib import pyplot as plt

config = {
    'algorithm': 'freematch',
    'net': 'Lite',
    'use_pretrain': False,
    'pretrain_path': 'C:/Users/kk/PycharmProjects/pytorchProject/Semi-supervised-learning-main/saved_models/fixmatch/model_best.pth',

    # optimization configs
    'epoch': 200,  # set to 100
    'num_train_iter': 200000,  # set to 102400
    'num_eval_iter': 1000,   # set to 1024
    'num_log_iter': 200,    # set to 256
    'optim': 'Adam',
    'lr': 1e-4,
    'amp': False,
    'weight_decay': 0,
    'layer_decay': 0,
    'batch_size': 16,
    'eval_batch_size': 16,

    # dataset configs
    'dataset': 'mnist',
    'num_labels': 4096,
    'num_classes': 4096,
    # 'img_size': 4096,
    'crop_ratio': 0.875,
    'data_dir': './data',

    # algorithm specific configs
    'hard_label': True,
    'uratio': 1,
    'ulb_loss_ratio': 1.0,

    # device configs
    'gpu': 0,
    'world_size': 1,
    "num_workers": 1,
    'distributed': False,

    # 其他
    'seed': 2024,
    'save_dir': './saved_models_free_v2.ms40_saf/classic_cv',
    'save_name': 'freematch',
    'load_path': './saved_models_free_v2.ms40_saf/classic_cv/freematch/latest_model.pth',
    'T': 0.5,

}
# config = get_config(config)
if __name__ == '__main__':
    config = get_config(config)  # 创建配置
    algorithm = get_algorithm(config, get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

    getdata = GetData('D:/data_1w')
    data, label = getdata.get_data()
    train_data, val_data, train_label, val_label = train_test_split(data, label, test_size=0.2, random_state=1)

    lb_data, lb_target, ulb_data, ulb_target = split_ssl_data(config, train_data, train_label, 4096,
                                                              config.num_labels,
                                                              include_lb_to_ulb=False)

    lb_dataset = GetDataset(lb_data, lb_target, is_ulb=False)
    ulb_dataset = GetDataset(ulb_data, ulb_target, is_ulb=True)



    eval_dataset = GetDataset(val_data, val_label, is_ulb=False)

    train_lb_loader = get_data_loader(config, lb_dataset, config.batch_size, shuffle=True, pin_memory=False)
    train_ulb_loader = get_data_loader(config, ulb_dataset, int(config.batch_size * config.uratio), shuffle=True,
                                       pin_memory=False)
    eval_loader = get_data_loader(config, eval_dataset, config.eval_batch_size, shuffle=True, pin_memory=False)

    trainer = Trainer(config, algorithm)
    trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)
    # trainer.evaluate(eval_loader, True)