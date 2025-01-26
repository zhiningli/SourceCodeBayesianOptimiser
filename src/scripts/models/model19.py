model = """
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.dropout2(x)
        
        x = self.fc4(x)
        return x

"""

corrupted_model = """
class Model(nn.Module):
    def __init__(self, feature_dim, class_count):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(feature_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, class_count)
        self.activate = nn.ReLU()
        self.dropout_a = nn.Dropout(p=0.3)
        self.dropout_b = nn.Dropout(p=0.5)

    def forward(self, input_tensor):
        x = self.layer1(input_tensor)
        x = self.activate(x)
        x = self.dropout_a(x)
        x = self.layer2(x)
        x = self.activate(x)
        x = self.layer3(x)
        x = self.dropout_b(x)
        x = self.output_layer(x)
        return x

"""