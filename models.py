import torch
import torch.nn as nn
import torch.nn.functional as f
import time
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_kernels, device,
                 num_face_features=12,
                 num_classes=2,
                 num_faces=9000,
                 num_neighbours=7
                 ):
        super(CNN, self).__init__()
        self.name = "CNN"
        self.post_conv_size = num_faces * num_kernels

        self.conv1 = nn.Conv2d(in_channels=num_face_features,
                               out_channels=num_kernels,
                               kernel_size=(1, num_neighbours), bias=False)
        self.conv2 = nn.Conv2d(in_channels=num_kernels,
                               out_channels=num_kernels,
                               kernel_size=(1, num_neighbours), bias=False)
        self.conv3 = nn.Conv2d(in_channels=num_kernels,
                               out_channels=num_kernels,
                               kernel_size=(1, num_neighbours), bias=False)

        self.fc1 = nn.Linear(self.post_conv_size, 100)
        self.bnfc1 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(100, 30)
        self.bnfc2 = nn.BatchNorm1d(30)

        self.fc_out = nn.Linear(30, num_classes)
        self.device = device

    def forward(self, x, adjacencies):
        # x: (64, 12, 9000, 7)
        # adjacencies: (64, 1, 9000, 7)

        x = self.conv1(x)
        # (64, 5, 9000, 1)
        x = self.redo_adjacencies(x, adjacencies)
        # (64, 5, 9000, 7)

        x = self.conv2(x)
        # (64, 5, 9000, 1)
        x = self.redo_adjacencies(x, adjacencies)
        # (64, 5, 9000, 7)

        x = self.conv3(x)
        # (64, 5, 9000, 1)

        x = x.squeeze()
        # (64, 5, 9000)
        x = x.reshape((x.shape[0], self.post_conv_size))
        # (64, 45000)

        x = self.relu(self.bnfc1(self.fc1(x)))
        x = self.relu(self.bnfc2(self.fc2(x)))
        x = self.fc_out(x)

        return x

    def redo_adjacencies(self, x, adjacencies):
        # x: (64, 5, 9000, 1)
        # adjacencies: (64, 1, 9000, 7)
        num_kernels = x.shape[1]
        num_neighbours = adjacencies.shape[3]

        x = torch.stack([x] * num_neighbours, dim=3).squeeze()
        # x: (64, 5, 9000, 7)
        adjacencies = torch.stack([adjacencies] * num_kernels, dim=1).squeeze().long()
        # adjacencies: (64, 5, 9000, 7)
        x = torch.gather(x, 2, adjacencies)
        # x: (64, 5, 9000, 7)
        return x



