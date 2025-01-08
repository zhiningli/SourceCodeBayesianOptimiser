dataset = """
# Dataset 5: Balanced classes with moderate complexity
X5, y5 = make_classification(
    n_samples=4000,
    n_features=30,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    weights=[0.33, 0.33, 0.34],  # Balanced class weights
    class_sep=1.2,
    flip_y=0.05,
    random_state=5
)
X5 = torch.tensor(X5, dtype=torch.float32)
y5 = torch.tensor(y5, dtype=torch.long)
train_dataset = TensorDataset(X5, y5)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Reusing the same dataset for simplicity
"""