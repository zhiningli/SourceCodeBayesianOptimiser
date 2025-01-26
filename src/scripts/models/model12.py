model = """
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.05)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        
        x = self.fc2(x)
        x = self.leaky_relu(x)
        
        x = self.fc3(x)
        return x

"""


corrupted_model = """
class Model(nn.Module):  # Neural network model definition
    def __init__(self, input_dim, output_classes):
        # Initialize the parent class
        super(Model, self).__init__()
        
        # Define the layers of the network
        self.layer1 = nn.Linear(input_dim, 128)  # First layer
        self.layer2 = nn.Linear(128, 64)  # Second layer
        self.output_layer = nn.Linear(64, output_classes)  # Output layer
        
        # Define the activation function
        self.activation_function = nn.LeakyReLU(negative_slope=0.05)  # Leaky ReLU activation

    # Forward pass through the network
    def forward(self, input_tensor):
        # Apply the first layer and activation
        out = self.layer1(input_tensor)
        out = self.activation_function(out)
        
        # Apply the second layer and activation
        out = self.layer2(out)
        out = self.activation_function(out)
        
        # Apply the final output layer
        out = self.output_layer(out)
        return out
"""