import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia as K
import typing
from typing import Tuple, List
from PIL import Image
import os
from tqdm import tqdm_notebook as tqdm
from time import time
import torchvision as tv


class Unet(nn.Module):
    """"""

    def __init__(self):
        super(Unet, self).__init__()
        return

    def forward(self, input):
        out = input
        return out


def get_downscale_idxs(model: nn.Module):
    downscale_idxs = []
    for i, l in enumerate(model.features):
        try:
            stride = l.stride
        except:
            stride = 1
        if type(stride) is tuple:
            stride = stride[0]
        if stride > 1:
            downscale_idxs.append(i)
    return downscale_idxs


class UnetFromPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(self.init_encoder())
        self.decoder_blocks = nn.ModuleList(self.init_decoder())
        return

    def forward(self, x):
        with torch.no_grad():
            skip_connections = []
            for block in self.encoder_blocks[:-1]:
                x = block(x)
                skip_connections.append(x.clone())

            x = self.encoder_blocks[-1](x)
        x = self.decoder_blocks[0](x)

        for block, skip_connection in zip(self.decoder_blocks[1:], reversed(skip_connections)):
            x = torch.cat([x, skip_connection], dim=1)
            x = block(x)
        return x

    def parameters(self):
        for block in self.encoder_blocks:
            for p in block.parameters():
                yield p
        for block in self.decoder_blocks:
            for p in block.parameters():
                yield p

    def init_encoder(self):
        vgg = tv.models.vgg13_bn(True)
        downscale_ids = get_downscale_idxs(vgg)
        downscale_ids = [-1] + downscale_ids
        encoder_blocks = []
        for index in range(len(downscale_ids) - 1):
            block = vgg.features[downscale_ids[index] + 1 : downscale_ids[index + 1]]
            if index != 0:
                # block.append(K.geometry.Rescale(antialias=True, factor=0.5))
                block = nn.Sequential(K.geometry.Rescale(antialias=True, factor=0.5), *block)
            encoder_blocks.append(nn.Sequential(*block))
        return encoder_blocks

    def init_decoder(self):
        decoder_dims = [1024, 512, 256, 128]
        decoder_blocks = [
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                # nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                K.geometry.Rescale(factor=(2, 2), antialias=True),
            )
        ]
        for index, dim in enumerate(decoder_dims):
            block = []
            block.append(nn.Conv2d(dim, dim // 2, padding=1, kernel_size=3))
            block.append(nn.BatchNorm2d(dim // 2))
            # block.append(nn.GroupNorm(16, dim // 2))
            block.append(nn.ReLU(inplace=True))
            block.append(nn.Conv2d(dim // 2, dim // 2, padding=1, kernel_size=3))
            block.append(nn.BatchNorm2d(dim // 2))
            # block.append(nn.GroupNorm(16, dim // 2))
            block.append(nn.ReLU(inplace=True))
            if index != len(decoder_dims) - 1:
                block.append(nn.Conv2d(dim // 2, dim // 4, kernel_size=1))
                block.append(nn.BatchNorm2d(dim // 4))
                # block.append(nn.GroupNorm(16, dim // 4))
                block.append(nn.ReLU(inplace=True))
                block.append(K.geometry.Rescale(factor=(2, 2), antialias=True))
            else:
                block.append(nn.Conv2d(dim // 2, 3, kernel_size=1))
            decoder_blocks.append(nn.Sequential(*block))
        return decoder_blocks


class ContentLoss(nn.Module):
    """"""

    def __init__(self, device=None, layer_id=11):
        super().__init__()
        # alex_net = tv.models.alexnet(True)
        with torch.no_grad():
            self.model = nn.Sequential(*tv.models.alexnet(True).features[:layer_id])
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, input, label):
        # loss = input.mean()
        y_pred = self.model(input)
        y = self.model(label)
        return self.mse_loss(y, y_pred)
        # return loss#

