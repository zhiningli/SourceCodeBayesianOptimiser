model = """
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        x = self.fc5(x)
        return x

"""

corrupted_model = """
class Model(nn.Module):
    def __init__(self, feature_dim, output_classes):
        super(Model, self).__init__()
        self.layer_1 = nn.Linear(feature_dim, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.final_layer = nn.Linear(64, output_classes)
        self.activation = nn.ReLU()

    def forward(self, input_data):
        out = self.layer_1(input_data)
        out = self.activation(out)
        
        out = self.layer_2(out)
        out = self.activation(out)
        
        out = self.layer_3(out)
        out = self.activation(out)
        
        out = self.layer_4(out)
        out = self.final_layer(out)
        return out

"""