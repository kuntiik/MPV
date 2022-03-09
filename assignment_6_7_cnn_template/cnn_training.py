from itertools import count
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia as K
import typing
from typing import Tuple, List, Dict
from PIL import Image
import os

# import deepcopkj

# import wandb

from pathlib import Path

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from time import time


def get_dataset_statistics(dataset: torch.utils.data.Dataset) -> Tuple[List, List]:
    """Function, that calculates mean and std of a dataset (pixelwise)
    Return:
        tuple of Lists of floats. len of each list should equal to number of input image/tensor channels
    """
    img_sum, img_std = 0, 0
    for img, _ in dataset:
        img_sum += torch.mean(img, dim=[1, 2])
        img_std += torch.mean(img ** 2, dim=[1, 2])
    mean = img_sum / len(dataset)
    std = (img_std / len(dataset) - mean ** 2) ** 0.5

    return mean, std


class SimpleCNN(nn.Module):
    """Class, which implements image classifier. """

    def __init__(self, num_classes=10):
        # super(SimpleCNN, self).__init__()
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes)
        )
        return

    def forward(self, input):
        """ 
        Shape:
        - Input :math:`(B, C, H, W)` 
        - Output: :math:`(B, NC)`, where NC is num_classes
        """
        x = self.features(input)
        return self.clf(x)


def weight_init(m: nn.Module) -> None:
    """Function, which fills-in weights and biases for convolutional and linear layers"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
    return


def train_and_val_single_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epoch_idx=0,
    lr_scheduler=None,
    writer=None,
    device: torch.device = torch.device("cpu"),
    additional_params: Dict = {},
) -> torch.nn.Module:
    """Function, which runs training over a single epoch in the dataloader and returns the model. Do not forget to set the model into train mode and zero_grad() optimizer before backward."""
    model = model.to(device)
    # for idx, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):

    model, train_loss, train_acc = train_epoch(model, train_loader, optim, loss_fn, device)
    val_loss, additional_out = validate_epoch(
        model, val_loader, loss_fn, device, additional_params
    )
    # TODO uncoment for tensorboard
    if writer is not None:
        writer.add_scalar("Accuracy/train", train_acc, epoch_idx)
        writer.add_scalar("Accuracy/val", additional_out["acc"], epoch_idx)
        writer.add_scalar("Loss/val", val_loss, epoch_idx)
        writer.add_scalar("Loss/train", train_loss, epoch_idx)
    # TODO uncoment for wandb logs
    # wandb.log({"Accuracy/train": train_acc.item(), "epoch": epoch_idx}, step=epoch_idx)
    # wandb.log({"Accuracy/val": additional_out["acc"].item()}, step=epoch_idx)
    # wandb.log({"Loss/train": train_loss}, step=epoch_idx)
    # wandb.log({"Loss/val": val_loss}, step=epoch_idx)
    return model


def lr_find(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    min_lr: float = 1e-7,
    max_lr: float = 100,
    steps: int = 50,
) -> Tuple:
    """Function, which run the training for a small number of iterations, increasing the learning rate and storing the losses. Model initialization is saved before training and restored after training"""
    model_orig = copy.deepcopy(model)
    learning_rates = np.logspace(np.log10(min_lr), np.log10(max_lr), steps)
    losses = []
    train_dl_iter = iter(train_dl)
    avg_loss = 0
    beta = 0.8

    for idx, lr in enumerate(learning_rates):
        xb, yb = next(train_dl_iter)
        prediction = model(xb)
        loss = loss_fn(prediction, yb)
        loss.backward()
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        optim.step()
        optim.zero_grad()
        avg_loss = (1 - beta) * loss.item() + beta * avg_loss
        smoothed_loss = avg_loss / (1 - beta ** (idx + 1))
        losses.append(smoothed_loss)

    lrs = learning_rates
    model = model_orig
    return losses, lrs


def accuracy(preds, yb):
    preds_index = preds.argmax(-1)
    return (preds_index == yb).float().mean()


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    additional_params: Dict = {},
) -> Tuple[float, Dict]:
    """Function, which runs the module over validation set and returns accuracy"""
    # print("Starting validation")
    acc = 0
    loss = 0
    count = 0
    model.train()
    model = model.to(device)
    for idx, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader), leave=True):
        data = data.to(device)
        labels = labels.to(device)
        preds = model(data)
        acc += accuracy(preds, labels) * data.size(dim=0)
        count += data.size(dim=0)
        loss_tmp = loss_fn(preds, labels)
        loss_tmp.backward()
        optim.step()
        optim.zero_grad()
        loss += loss_tmp.item()

    loss /= count
    acc /= count

    return model, loss, acc


# This is a replacement for validate
def validate_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    additional_params: Dict = {},
) -> Tuple[float, Dict]:
    """Function, which runs the module over validation set and returns accuracy"""
    # print("Starting validation")
    acc = 0
    loss = 0
    count = 0

    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for idx, (data, labels) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            data = data.to(device)
            labels = labels.to(device)
            preds = model(data)
            acc += accuracy(preds, labels) * data.size(dim=0)
            count += data.size(dim=0)
            loss += loss_fn(preds, labels).item()
        loss /= count
        acc /= count
        print("Validation acc : ", acc)

    return loss, {"acc": acc}


class TestFolderDataset(torch.utils.data.Dataset):
    """Class, which reads images in folder and serves as test dataset"""

    def __init__(self, folder_name, transform=None):
        root = Path(folder_name)
        self.fnames = list(root.iterdir())
        self.transfroms = transform

    def __getitem__(self, index):
        # img = Image.new("RGB", (128, 128))
        img = Image.open(self.fnames[index])
        if self.transfroms is not None:
            img = self.transfroms(img)
        return img

    def __len__(self):
        return len(self.fnames)


def get_predictions(model: torch.nn.Module, test_dl: torch.utils.data.DataLoader) -> torch.Tensor:
    """Function, which predicts class indexes for image in data loader. Ouput shape: [N, 1], where N is number of image in the dataset"""
    # out = torch.zeros(len(test_dl)).long()
    model.eval()
    out = []
    with torch.no_grad():
        for idx, xb in tqdm(enumerate(test_dl), total=len(test_dl), leave=False):
            preds = model(xb)
            preds_index = preds.argmax(-1)
            out.append(preds_index)
    out = torch.cat(out)
    return out
