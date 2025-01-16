extract_information_from_script_prompts = """
Your task is to analyze and extract relevant sections from the provided source code.
<code>
{source_code}
</code> 
Please help me extract the following information:

1. "The section that contains the model architecture, retain comments if any"
2. "The section that contains the dataset information"

Here is an example of how this should be done. Pay attention to the <> used as it will be used for parsing later "

```python
<model>
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        
        # Reshape the input to (batch_size, 1, input_size) for Conv1d
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (input_size // 2), 128)  # Adjust for the reduced dimension after pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add regularization with dropout

    def forward(self, x):
        # Add a channel dimension for Conv1d (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
</model>

<dataset>
X, y = make_classification(
    n_samples=4000,          # Number of samples
    n_features=25,           # Total number of features
    n_informative=10,        # Number of informative features
    n_redundant=2,           # Number of redundant features
    n_classes=3,             # Number of classes
    n_clusters_per_class=2,  # Increase clusters per class for complexity
    class_sep=0.5,           # Reduce class separation for overlap
    flip_y=0.1,              # Introduce noise in labels
    random_state=42          # Random state for reproducibility
)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Reusing the same dataset for simplicity
</dataset>

Notes:
1. Ensure only the relevant sections are extracted and formatted exactly as shown above.
2. If a section is missing, indicate it like this

<model>
# No model architecture found
</model>

3. Sometimes models are consisted of multiple classes, extract them together like follows:
<model>
class ResidualBlock(nn.Module):
    def __init__(self, input_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size, input_size)

    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x += residual  # Skip connection
        return x

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.fc2(x)
        return x
</model>
4. Ensure the extracted sections are properly formatted and adhere to Python syntax.
5. Do not include any additional explanations or comments outside the required sections. 

Return your findings as a python script like above:
<model>
(the model that used in the code str)
</model>

<dataset>
(the dataset that used in the code str)
</dataset>
"""