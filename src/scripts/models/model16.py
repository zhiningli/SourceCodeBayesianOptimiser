model = """
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = torch.max(x1, x2)
        x = self.fc3(x)
        return x

"""

corrupted_model = """
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        





        
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = torch.max(x1, x2)
        x = self.fc3(x)
        return x

"""