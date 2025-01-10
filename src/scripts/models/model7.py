model = """
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  # Hidden layer with 32 neurons
        self.fc2 = nn.Linear(32, num_classes) # Output layer
        self.relu = nn.ReLU()                 # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU after the first layer
        x = self.fc2(x)             # Output layer (no activation for logits)
        return x

"""