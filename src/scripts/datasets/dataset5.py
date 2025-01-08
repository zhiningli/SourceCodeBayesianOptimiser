dataset = """
# Dataset 4: Low-dimensional with high overlap
X4, y4 = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=5,
    class_sep=0.3,
    flip_y=0.15,
    random_state=4
)
X4 = torch.tensor(X4, dtype=torch.float32)
y4 = torch.tensor(y4, dtype=torch.long)
train_dataset = TensorDataset(X4, y4)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Reusing the same dataset for simplicity
"""