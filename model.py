# -*- coding: UTF-8 -*-
'''
@Project : trytry 
@File    : model.py
@Author  : kliu
@Date    : 2023/10/29
'''

import torch
import torch.nn as nn
import torch.optim as optim
import config
from util import seed_everything, mean_rmse, split, load_checkpoints, save_checkpoints
from tqdm import tqdm
import os
import pandas as pd


class AE(nn.Module):
    def __init__(self, load: bool = False, hidden_size: int = 18211, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.load = load
        self.model = nn.Sequential(
            nn.Linear(hidden_size, 10000),
            nn.BatchNorm1d(10000),
            nn.Tanh(),
            nn.Linear(10000, 4096),
            nn.BatchNorm1d(4096),
            nn.Tanh(),
            nn.Linear(4096, 1024),
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.Tanh(),
            nn.Linear(4096, 10000),
            nn.BatchNorm1d(10000),
            nn.Tanh(),
            nn.Linear(10000, hidden_size)
        )

    def forward(self, x):
        return self.model(x)
    
    def train(self, data, label):
        # initialization
        model = self.model
        optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.1)
        data = torch.tensor(data, dtype=torch.float32).to(device=config.DEVICE)
        label = torch.tensor(label, dtype=torch.float32).to(device=config.DEVICE)

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        seed_everything(42)

        if self.load:
            checkpoints = load_checkpoints(config.CHECKPOINT)
            model.load_state_dict(checkpoints["model"])
            optimizer.load_state_dict(checkpoints["optimizer"])

        loss_all = []
        loop = tqdm(range(config.EPOCHS))
        for epoch in loop:
            # forward
            out = model(data)
            loss = mean_rmse(out, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            # early stopping
            loss_all.append(loss)
            loop.set_description(f"[Epoch {epoch + 1}/{config.EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # 4. save the model
        if epoch % 10 == 0:
            file = os.path.join(config.CHECKPOINT, f"checkpoint_{epoch}.pth.tar")
            save_checkpoints(model, optimizer, pth=file)

    def test(self, test_data):
        self.model.eval()
        output_test = self.model(test_data).cpu().detach().numpy()
        return output_test