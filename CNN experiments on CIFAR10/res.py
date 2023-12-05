import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation, kernel_size=3, pooling=1):
        super().__init__()
        layers = [
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=int(kernel_size / 2), bias=False),
            nn.BatchNorm2d(out_channel),
            activation(inplace=True)
        ]
        if pooling:
            layers.insert(1, nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResPoolBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation, pooling=1):
        super().__init__()
        layers = [
            nn.BatchNorm2d(in_channel),
            activation(inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            activation(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        ]
        if pooling:
            layers.insert(3, nn.MaxPool2d(2))
        self.side = nn.Sequential(*layers)
        self.main = nn.Identity() if not pooling and in_channel == out_channel else nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.side(x) + self.main(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation, stride=1):
        super().__init__()
        layers = [
            nn.BatchNorm2d(in_channel),
            activation(inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, stride+1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            activation(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        ]
        self.side = nn.Sequential(*layers)
        self.main = nn.Identity() if not stride and in_channel == out_channel else nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=stride+1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.side(x) + self.main(x)
        return out


class ResNet9(nn.Module):
    def __init__(self, activation, block=ResBlock, num_class=10):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, activation, pooling=0)
        self.conv2 = ConvBlock(64, 128, activation)
        self.res1 = block(128, 128, activation, 0)
        self.conv3 = ConvBlock(128, 256, activation)
        self.conv4 = ConvBlock(256, 512, activation)
        self.res2 = block(512, 512, activation, 0)
        self.fc = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.fc(x)
        return x


class ResNet12(nn.Module):
    def __init__(self, activation, block=ResBlock, num_class=10):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, activation, pooling=False)
        self.res1 = block(64, 64, activation, 0)
        self.conv2 = ConvBlock(64, 128, activation)
        self.res2 = block(128, 128, activation, 0)
        self.conv3 = ConvBlock(128, 256, activation)
        self.res3 = block(256, 256, activation, 0)
        self.conv4 = ConvBlock(256, 512, activation)
        self.res4 = block(512, 512, activation, 0)
        self.fc = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.res3(x)
        x = self.conv4(x)
        x = self.res4(x)
        x = self.fc(x)
        return x


class ResNet16(nn.Module):
    def __init__(self, activation, block=ResBlock, num_class=10):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, activation, pooling=False)
        self.res1 = block(64, 64, activation, 0)
        self.res2 = block(64, 128, activation, 1)
        self.res3 = block(128, 128, activation, 0)
        self.res4 = block(128, 256, activation, 1)
        self.res5 = block(256, 256, activation, 0)
        self.res6 = block(256, 512, activation, 1)
        self.res7 = block(512, 512, activation, 0)
        self.fc = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.fc(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, activation, block=ResBlock, num_class=10):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, activation, pooling=False)
        self.res1 = nn.Sequential(
            block(64, 64, activation, 0),
            block(64, 64, activation, 0)
        )
        self.res2 = nn.Sequential(
            block(64, 128, activation, 1),
            block(128, 128, activation, 0)
        )
        self.res3 = nn.Sequential(
            block(128, 256, activation, 1),
            block(256, 256, activation, 0)
        )
        self.res4 = nn.Sequential(
            block(256, 512, activation, 1),
            block(512, 512, activation, 0)
        )
        self.fc = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(30)
    net = ResNet12(nn.LeakyReLU)
    summary(net, (3, 32, 32))
