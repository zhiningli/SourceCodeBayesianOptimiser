from src.preprocessing.MLP_BO_optimiser import MLP_BO_Optimiser
import random
import json
import torch

code_str = """
import openml
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data.values  # Convert DataFrame to NumPy array
        self.labels = labels.values  # Convert Series to NumPy array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32)  # Reshape from flat array to CHW
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)  # Apply transformations
        return img, label


def to_tensor(image):
    return torch.tensor(image, dtype=torch.float32) / 255.0  # Scale to [0, 1]


def normalize(image, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)  # Reshape for broadcasting
    std = torch.tensor(std).view(-1, 1, 1)
    return (image - mean) / std


def preprocess(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    tensor = to_tensor(image)
    return normalize(tensor, mean, std)


def load_openml_cifar10():
    # Load dataset as a DataFrame
    dataset = openml.datasets.get_dataset(dataset_id=40927)  # CIFAR-10 dataset ID

    # Get data and ensure target is included
    df, _, _, _ = dataset.get_data(dataset_format="dataframe")


    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)  # Exclude target column
    y = df.iloc[:, -1].to_numpy(dtype=np.int64)  # Extract target column


    # Wrap features and labels into pandas DataFrame/Series
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y, name="label")

    # Split dataset into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(X_df))
    X_train = X_df.iloc[:train_size].reset_index(drop=True)
    y_train = y_series.iloc[:train_size].reset_index(drop=True)
    X_val = X_df.iloc[train_size:].reset_index(drop=True)
    y_val = y_series.iloc[train_size:].reset_index(drop=True)

    return X_train, y_train, X_val, y_val




def run_mlp_classification(hidden1, hidden2, hidden3, hidden4, activation, lr, weight_decay, epoch):
    # Load CIFAR-10 dataset from OpenML
    X_train, y_train, X_val, y_val = load_openml_cifar10()

    # Define transformations
    def transform(image):
        return preprocess(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # Create PyTorch datasets
    train_dataset = CIFAR10Dataset(X_train, y_train, transform=transform)
    val_dataset = CIFAR10Dataset(X_val, y_val, transform=transform)

    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model definition
    input_size = 3 * 32 * 32  # CIFAR-10 images are 3x32x32
    activation_fn = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}[activation]
    model = nn.Sequential(
        nn.Flatten(),  # Flatten the input tensor
        nn.Linear(input_size, hidden1), activation_fn(),
        nn.Linear(hidden1, hidden2), activation_fn(),
        nn.Linear(hidden2, hidden3), activation_fn(),
        nn.Linear(hidden3, hidden4), activation_fn(),
        nn.Linear(hidden4, 10)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Train model
    model.train()
    for epoch in range(epoch):  # Small number of epochs for optimization speed
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    return accuracy

"""


optimiser = MLP_BO_Optimiser()

results_for_plotting = {}

hyperparameters = {
    "MLP_hidden1_nu": [0.5, 1.5, 2.5],
    "MLP_hidden2_nu": [0.5, 1.5, 2.5],
    "MLP_hidden3_nu": [0.5, 1.5, 2.5],
    "MLP_hidden4_nu": [0.5, 1.5, 2.5],
    "MLP_lr_nu": [0.5, 1.5, 2.5],
    "MLP_activation_nu": [0.5, 1.5, 2.5],
    "MLP_weight_decay_nu": [0.5, 1.5, 2.5],
}

results_for_plotting = {}  # Dictionary to store results

for i in range(20):  # Loop for 20 sets of hyperparameters
    print(f"Running {i + 1} set of BO hyperparameters")
    
    # Randomly sample hyperparameter indices and retrieve their corresponding values
    hyper_indices = [random.randint(0, 2) for _ in range(7)]  # Generate 7 random indices for hyperparameters
    hyper_values = [
        hyperparameters["MLP_hidden1_nu"][hyper_indices[0]],
        hyperparameters["MLP_hidden2_nu"][hyper_indices[1]],
        hyperparameters["MLP_hidden3_nu"][hyper_indices[2]],
        hyperparameters["MLP_hidden4_nu"][hyper_indices[3]],
        hyperparameters["MLP_lr_nu"][hyper_indices[4]],
        hyperparameters["MLP_activation_nu"][hyper_indices[5]],
        hyperparameters["MLP_weight_decay_nu"][hyper_indices[6]],
    ]
    
    print(f"Selected hyperparameter values: {hyper_values}")
    
    # Run the optimization for the selected hyperparameters
    accuracies, best_y, best_candidate = optimiser.optimise(
        code_str=code_str,
        MLP_hidden1_nu=hyper_values[0],
        MLP_hidden2_nu=hyper_values[1],
        MLP_hidden3_nu=hyper_values[2],
        MLP_hidden4_nu=hyper_values[3],
        MLP_lr_nu=hyper_values[4],
        MLP_activation_nu=hyper_values[5],
        MLP_weight_decay_nu=hyper_values[6],
    )
    
    # Store results for plotting
    results_for_plotting[tuple(hyper_values)] = {
        "accuracies": accuracies,
        "best_y": best_y,
        "best_candidate": best_candidate.tolist() if isinstance(best_candidate, torch.Tensor) else best_candidate,
    }
    print(f"Best Y: {best_y}, Best Candidate: {best_candidate}")

# Export results to a JSON file
output_file = "bo_results.json"
with open(output_file, "w") as f:
    json.dump(results_for_plotting, f, indent=4)

print(f"Results saved to {output_file}")

