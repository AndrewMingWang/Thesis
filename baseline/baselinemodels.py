import torch
import torch.nn as nn
import torch.nn.functional as f

class BaselineCNN(nn.Module):
    def __init__(self, num_kernels, num_classes=10):
        super(BaselineCNN, self).__init__()
        self.name = "BaselineCNN"

        self.conv1 = nn.Conv3d(1, num_kernels, kernel_size=(3, 3, 3))
        self.conv2 = nn.Conv3d(num_kernels, num_kernels, kernel_size=(3, 3, 3))
        self.conv3 = nn.Conv3d(num_kernels, num_kernels, kernel_size=(1, 3, 3))

        # TODO: Set this
        self.post_conv_size = num_kernels * 14 * 14

        self.fc1 = nn.Linear(self.post_conv_size, 100)
        self.bnfc1 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(100, 30)
        self.bnfc2 = nn.BatchNorm1d(30)

        self.fc_out = nn.Linear(30, num_classes)

    def forward(self, x):
        #print(x.shape)
        x = x.permute(0, 3, 1, 2)
        x = torch.unsqueeze(x, 1)
        #print(x.shape)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        #print(x.shape)
        x = x.squeeze()

        x = x.reshape((x.shape[0], self.post_conv_size))

        x = self.relu(self.bnfc1(self.fc1(x)))

        x = self.relu(self.bnfc2(self.fc2(x)))

        x = self.fc_out(x)

        return x





