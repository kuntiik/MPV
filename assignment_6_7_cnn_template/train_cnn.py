import torchvision.transforms as tfms
import matplotlib.pyplot as plt

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

# import kornia as K
from tqdm import tqdm_notebook as tqdm
from time import time

from cnn_training import SimpleCNN, train_and_val_single_epoch
from torch.utils.tensorboard import SummaryWriter

from efficientnet_pytorch import EfficientNet
from pathlib import Path
import wandb


print(Path.cwd())


def get_loaders(train_transform, val_transform, bs=32, num_workers=8):

    ImageNette_train = tv.datasets.ImageFolder(
        "/home.stud/kuntluka/MPV/assignment_6_7_cnn_template/imagenette2-160/train",
        transform=train_transform,
    )
    ImageNette_val = tv.datasets.ImageFolder(
        "/home.stud/kuntluka/MPV/assignment_6_7_cnn_template/imagenette2-160/val",
        transform=val_transform,
    )

    train_dl = torch.utils.data.DataLoader(
        ImageNette_train,
        batch_size=32,
        shuffle=True,  # important thing to do for training.
        num_workers=num_workers,
    )
    val_dl = torch.utils.data.DataLoader(
        ImageNette_val, batch_size=32, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_dl, val_dl


def get_transforms():
    mean, std = [0.46248055, 0.4579692, 0.42981696], [0.27553096, 0.27220666, 0.295335]
    img_size = (224, 224)
    train_transform = tfms.Compose(
        [
            tfms.Resize(img_size),
            # tfms.RandomHorizontalFlip(),
            # tfms.ColorJitter(0.1,0.1,0.1),
            tfms.ToTensor(),
            tfms.Normalize(mean, std),
        ]
    )

    val_transform = tfms.Compose(
        [tfms.Resize(img_size), tfms.ToTensor(), tfms.Normalize(mean, std)]
    )

    return train_transform, val_transform


def train(n_epochs=20):
    train_transforms, val_transforms = get_transforms()
    train_dl, val_dl = get_loaders(train_transforms, val_transforms)
    model = EfficientNet.from_pretrained("efficientnet-b0")

    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter("effnet")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-2)

    for epoch in range(n_epochs):
        model = train_and_val_single_epoch(
            model,
            train_dl,
            val_dl,
            opt,
            loss_fn,
            epoch_idx=epoch,
            writer=writer,
            device=torch.device("cuda"),
        )
    torch.save(model.state_dict(), "effnet_model.pt")

    # num_classes =
    # model


if __name__ == "__main__":
    wandb.login()
    wandb.init(project="mpv")
    train(7)
