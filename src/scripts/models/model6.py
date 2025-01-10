model = """
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, 256)       # First hidden layer
        self.bn1 = nn.BatchNorm1d(256)             # Batch normalization
        self.fc2 = nn.Linear(256, 128)             # Second hidden layer
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)              # Third hidden layer
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)      # Output layer
        
        # Activations
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
        
        # Dropout
        self.dropout = nn.Dropout(p=0.3)           # Regularization
        
    def forward(self, x):
        # Layer 1 with ReLU
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2 with LeakyReLU
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Layer 3 with Tanh
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.tanh(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc4(x)
        return x
"""