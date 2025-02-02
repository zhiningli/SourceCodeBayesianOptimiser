model = """
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        
        x = self.fc2(x)
        x = self.tanh(x)
        
        x = self.fc3(x)
        return x

"""

corrupted_model = """
class Model(nn.Module):  # Model with dropout regularization
    def __init__(self, input_dim, output_classes):
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_classes)
        
        self.activation_function = nn.Tanh()

    def forward(self, data):
        output = self.fc1(data)
        output = self.activation_function(output)
        
        output = self.fc2(output)
        output = self.activation_function(output)
        
        output = self.fc3(output)
        return output
"""
