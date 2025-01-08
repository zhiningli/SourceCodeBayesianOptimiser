dataset = """
# Dataset 1: Basic Binary Classification
X1, y1 = make_classification(
    n_samples=3000,
    n_features=20,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    class_sep=1.0,
    flip_y=0.01,
    random_state=1
)
X1 = torch.tensor(X1, dtype=torch.float32)
y1 = torch.tensor(y1, dtype=torch.long)
train_dataset = TensorDataset(X1, y1)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Reusing the same dataset for simplicity
"""