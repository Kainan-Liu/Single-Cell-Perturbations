import numpy as np
import torch
import torch.nn as nn
import random
import pandas as pd
import matplotlib.pyplot as plt



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
    datatype = type(y_pred)
    if datatype == torch.Tensor:
        mse = torch.mean(torch.square(y_pred - y_true), dim=1)
        mrrmse = torch.mean(torch.sqrt(mse))
    else:
        mse = np.mean(np.square(y_pred - y_true))
        mrrmse = np.sqrt(mse)
    return mrrmse


def submit_kaggle(output, submit_time):
    sample_submission = pd.read_csv("./data/sample_submission.csv")
    my_submission = pd.concat([sample_submission.iloc[:,0], output], axis=1)
    my_submission.to_csv(f"submission_{submit_time}.csv", index=False)


def split(output, split_size, i, *, lgb=True):
    """split the output"""
    output_size = output.shape[1] // split_size if lgb else output.shape[1] // split_size + 1
    sub_output = output[output.columns[i * output_size:(i + 1) * output_size]]
    column_name = sub_output.columns.tolist()
    return sub_output,column_name

def plot_loss(loss_list, method):
    plt.plot(loss_list, label='Loss')
    plt.title(f"{method}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()