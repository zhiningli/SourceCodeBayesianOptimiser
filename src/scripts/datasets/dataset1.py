dataset = """
# Generate a dataset with 5 classes
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
"""