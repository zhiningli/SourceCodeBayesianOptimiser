dataset = """
from sklearn.datasets import load_breast_cancer

# Load Breast Cancer dataset
bc = load_breast_cancer()
X8 = torch.tensor(bc.data, dtype=torch.float32)  # Features
y8 = torch.tensor(bc.target, dtype=torch.long)  # Labels

# Convert to TensorDataset and DataLoader
train_dataset = TensorDataset(X8, y8)
train_loader= DataLoader(train_dataset8, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset8, batch_size=32, shuffle=True)

"""