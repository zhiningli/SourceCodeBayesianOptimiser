dataset = """
# Dataset 2: Multi-class with more overlap
X2, y2 = make_classification(
    n_samples=4000,
    n_features=25,
    n_informative=12,
    n_redundant=3,
    n_classes=3,
    class_sep=0.5,
    flip_y=0.1,
    random_state=2
)
X2 = torch.tensor(X2, dtype=torch.float32)
y2 = torch.tensor(y2, dtype=torch.long)
train_dataset = TensorDataset(X2, y2)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Reusing the same dataset for simplicity
"""