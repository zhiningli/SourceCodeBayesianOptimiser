dataset = """
X8, y8 = make_classification(
    n_samples=2000,
    n_features=50,
    n_informative=30,
    n_redundant=5,
    n_classes=3,
    class_sep=2.0,  # Highly separable
    flip_y=0.01,
    random_state=8
)
X8 = torch.tensor(X8, dtype=torch.float32)
y8 = torch.tensor(y8, dtype=torch.long)
train_dataset = TensorDataset(X8, y8)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Reusing the same dataset for simplicity
"""