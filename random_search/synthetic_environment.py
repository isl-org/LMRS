import torch
import torch.nn as nn
import torch.nn.functional as F

class SyntheticFunction(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, noise_level):
        super(SyntheticFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.noise_level = noise_level
        print('Created Function with {} {} {} {}'.format(input_dim, hidden_dim, output_dim, noise_level))

    def forward(self, x, with_noise=True):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        if with_noise:
            x = x + torch.randn(x.size()).cuda()*self.noise_level
        return (x - 1.0)*(x - 1.0)

class SyntheticFunctionLearner(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SyntheticFunctionLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        h = self.fc2(x)
        y = torch.relu(h)
        y = self.fc3(y)
        return y, h
