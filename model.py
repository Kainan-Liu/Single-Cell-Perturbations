# -*- coding: UTF-8 -*-
'''
@Project ：trytry 
@File    ：model.py
@Author  ：mhuang
@Date    ：2023/10/24
'''

import torch
import torch.nn as nn
import torch.optim as optim
import config
from util import seed_everything, mean_rmse, split, load_checkpoints, save_checkpoints
from tqdm import tqdm
import os
import pandas as pd

class nnRegression(nn.Module):
    """model 1"""
    def __init__(self, input_train, output_train):
        super(nnRegression,self).__init__()
        self.input_size = input_train.shape[1]
        self.output_size = output_train.shape[1]
        self.pred = nn.Sequential(nn.Linear(self.input_size, 64),  #放到高维空间好眼熟
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(),
                                     nn.Linear(64, 256),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Linear(512, self.output_size))

    def forward(self, x):
        x = x.float()  #x是int型数据
        x = self.pred(x)
        return x


class myRegression(nn.Module):
    def __init__(self, input_train, output_train, input_test, split_size, mode, submission=False):
        super(myRegression, self).__init__()

        self.input_train = torch.tensor(input_train.values)
        self.output_train = output_train
        self.input_test = torch.tensor(input_test.values)
        self.output_test = pd.DataFrame(torch.randn(size=(len(self.input_test), 0)))
        self.split_size = split_size
        self.mode = mode
        self.submission = submission

    def forward(self, epochs, load=False):
        loops=0

        for loops in tqdm(range(self.split_size)):
            # 1. data
            sub_output_train, sub_column_name = split(self.output_train, self.split_size, loops) #要在输入之前shuffle（如果需要）
            sub_output_train = torch.tensor(sub_output_train.values)

            # 2. initialize
            if self.mode == "nnRegression":
                self.model = nnRegression(self.input_train, sub_output_train)  # 这里的model有必要加self吗？

            model = self.model
            optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.1)  #可换
            # optimizer = optim.Adam(model.parameters(),
            #                        lr=1e-3,
            #                        betas=(0.1, 0.999))

            # scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
            #                                           max_lr=1,
            #                                           steps_per_epoch=len(sub_output_train),
            #                                           epochs=config.EPOCHS)

            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

            seed_everything(42)

            # 3. train
            if load:
                checkpoints = load_checkpoints(config.CHECKPOINT) #还要改的，路径不对，但是要想加载，也得考虑能不能用这个来直接预测test
                model.load_state_dict(checkpoints["model"])
                optimizer.load_state_dict(checkpoints["optimizer"])

            loss_all = []

            for epoch in range(config.EPOCHS):  #加个tqdm就可以看每个模型训练了多少次了
                # 3.1 forward
                sub_output_hat = model(self.input_train)
                loss = mean_rmse(sub_output_hat, sub_output_train)

                # 3.2 backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step()

                # 3.3 early stopping
                loss_all.append(loss)

                if epoch != 0 :
                    if abs(loss_all[-2] - loss) < 0.0001: #原文是如果在连续5个epoch中验证集上的损失值没有减少，就停止训练，并且保留在验证集上表现最好的模型的权重。
                         break

            # 4. save the model
            file = os.path.join(config.CHECKPOINT, f"checkpoint_{loops}.pth.tar")
            save_checkpoints(model, optimizer, pth=file)

            # 5. test_submission
            if self.submission == True:
                model.eval()
                sub_output_test = model(self.input_test)
                sub_output_test = sub_output_test.detach().numpy()
                sub_output_test = pd.DataFrame(sub_output_test, columns= sub_column_name)
                self.output_test = pd.concat((self.output_test, sub_output_test), axis=1)

        return loss_all, self.output_test