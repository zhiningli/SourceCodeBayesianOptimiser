model = """
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  
        self.fc2 = nn.Linear(32, num_classes) 
        self.relu = nn.ReLU()                

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU after the first layer
        x = self.fc2(x)             
        return x

"""