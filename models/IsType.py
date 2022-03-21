import torch
import torch.nn as nn


class IsType(nn.Module):
    def __init__(self, feature_size):
        super(IsType, self).__init__()
        self.feature_size = feature_size
        self.fc0 = nn.Linear(in_features=self.feature_size, out_features=128, bias=True)
        self.fc1 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc2 = nn.Linear(in_features=64, out_features=16, bias=True)
        self.fc3 = nn.Linear(in_features=16, out_features=4, bias=True)

    def forward(self, input):
        x = self.fc0(input)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
