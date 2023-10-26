import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import random

def save_checkpoints(model: nn.Module, optimizer: nn.Module, pth: str):
    #print("==> Saving Checkpoints")
    checkpoints = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoints, pth)


def load_checkpoints(pth: str):
    #print("==> Loading Checkpoints")
    checkpoints = torch.load(pth)
    return checkpoints

def seed_everything(random_state: int):
    """
    Make the results be reproducible
    :param random_state:用作种子
    :return: None
    """
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

def mean_rmse(y_pred, y_true):
    """calculate the Mean Rowwise Root Mean Squared Error """
    mse = torch.mean(torch.square(y_pred - y_true), dim=1)
    mrrmse = torch.mean(torch.sqrt(mse))
    return mrrmse

def squared_error(y_pred, y_true):
    """calculate the n*MSE（没用上） """
    return np.sum(np.square(y_pred - y_true), axis=1)

def split(output, split_size, i):
    """split the output"""
    output_size = output.shape[1] // split_size + 1
                 # 输出标签的总数量      #向上取整保证能够覆盖所有标签
    sub_output = output[output.columns[i * output_size:(i + 1) * output_size]]
    column_name = sub_output.columns.tolist()
    return sub_output,column_name
