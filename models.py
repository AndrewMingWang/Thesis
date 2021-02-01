import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self, num_kernels,
                 num_face_features=11,
                 num_classes=10,
                 num_faces=9000,
                 num_neighbours=7):
        super(CNN, self).__init__()
        self.name = "CNN"
        self.post_conv_size = num_faces * num_kernels

        self.conv1 = nn.Conv2d(in_channels=num_face_features,
                               out_channels=num_kernels,
                               kernel_size=(1, num_neighbours))
        self.bnconv1 = nn.BatchNorm2d(num_kernels)

        self.fc1 = nn.Linear(self.post_conv_size, 100)
        self.bnfc1 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(100, 30)
        self.bnfc2 = nn.BatchNorm1d(30)

        self.fc_out = nn.Linear(30, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = self.bnconv1(self.conv1(x))
        x = x.squeeze()
        x = x.reshape((x.shape[0], self.post_conv_size))

        x = self.relu(self.bnfc1(self.fc1(x)))

        x = self.relu(self.bnfc2(self.fc2(x)))

        x = self.fc_out(x)

        return x





