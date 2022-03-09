# inpired by : https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(Block, 64, num_blocks[0], stride=1)
        layer2 = self._make_layer(Block, 128, num_blocks[1], stride=2)
        layer3 = self._make_layer(Block, 256, num_blocks[2], stride=2)
        layer4 = self._make_layer(Block, 512, num_blocks[3], stride=2)
        self.features = nn.Sequential(conv1, bn1, nn.Relu(), layer1, layer2, layer3, layer4)
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        linear = nn.Linear(512, num_classes)
        self.clf = nn.Sequential(avgpool,nn.Flatten() linear)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.clf(x)
        return out
