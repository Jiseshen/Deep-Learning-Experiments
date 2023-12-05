import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary


class DenseLayer(nn.Module):
    def __init__(self, in_channel, growth, bn_size, drop):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, bn_size*growth, 1, bias=False),
            nn.BatchNorm2d(bn_size*growth),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size*growth, growth, 3, padding=1, bias=False)
        )
        if drop:
            self.layer = nn.Sequential(self.layer, nn.Dropout(drop))

    def forward(self, x):
        new = self.layer(x)
        return torch.cat([x, new], 1)


class DenseBlock(nn.Module):
    def __init__(self, num, in_channel, bn_size, growth, drop):
        super().__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channel + i * growth, growth, bn_size, drop) for i in range(num)])

    def forward(self, x):
        return self.layers(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    def __init__(self, growth=16, blocks=(3,6,12,8), first_channel=64, bn_size=4, drop=0, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, first_channel, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(first_channel),
            nn.ReLU(inplace=True)
        )
        dense_layers = []
        channel = first_channel
        for i, num in enumerate(blocks):
            dense_layers.append(DenseBlock(num, channel, bn_size, growth, drop))
            channel += num * growth
            if not i == len(blocks) - 1:
                dense_layers.append(ConvBlock(channel, channel//2))
                channel = channel // 2
        self.features = nn.Sequential(self.features,*dense_layers)
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(channel, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


if __name__ == '__main__':
    torch.manual_seed(30)
    myDenseNet = DenseNet()
    summary(myDenseNet, (3,32,32), 256)