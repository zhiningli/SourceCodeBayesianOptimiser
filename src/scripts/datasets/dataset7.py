dataset = """
# Dataset 6: Imbalanced classes
X6, y6 = make_classification(
    n_samples=4000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    n_classes=3,
    weights=[0.7, 0.2, 0.1],  # Imbalanced class weights
    class_sep=0.8,
    flip_y=0.1,
    random_state=6
)
X6 = torch.tensor(X6, dtype=torch.float32)
y6 = torch.tensor(y6, dtype=torch.long)
train_dataset = TensorDataset(X6, y6)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Reusing the same dataset for simplicity
"""