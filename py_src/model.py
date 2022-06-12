from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=512*7*7, out_features=1000)

    def forward(self, x: torch.Tensor):
        x = self.conv0(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv1(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv4(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv5(x)
        x = torch.tanh(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x
