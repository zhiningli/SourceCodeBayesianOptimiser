dataset = """
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()
X9 = torch.tensor(wine.data, dtype=torch.float32)  # Features
y9 = torch.tensor(wine.target, dtype=torch.long)  # Labels

# Convert to TensorDataset and DataLoader
train_dataset9 = TensorDataset(X9, y9)
train_loader = DataLoader(train_dataset9, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset9, batch_size=32, shuffle=True) 
"""