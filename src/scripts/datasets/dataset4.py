dataset = """
# Dataset 3: High-dimensional data with few samples
X3, y3 = make_classification(
    n_samples=1000,
    n_features=100,
    n_informative=20,
    n_redundant=10,
    n_classes=4,
    class_sep=0.8,
    flip_y=0.05,
    random_state=3
)
X3 = torch.tensor(X3, dtype=torch.float32)
y3 = torch.tensor(y3, dtype=torch.long)
train_dataset = TensorDataset(X3, y3)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Reusing the same dataset for simplicity
"""