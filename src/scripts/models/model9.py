model = """

import torch.nn.functional as F

class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_features, bottleneck_features=16):
        super(BottleneckResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, bottleneck_features) 
        self.fc2 = nn.Linear(bottleneck_features, bottleneck_features) 
        self.fc3 = nn.Linear(bottleneck_features, in_features) 
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x += residual  # Skip connection
        return F.relu(x)  # Apply activation to the output

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.res1 = BottleneckResidualBlock(256, 64)
        self.res2 = BottleneckResidualBlock(256, 64)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.fc2(x)
        return x

"""