from src.preprocessing.MLP_BO_optimiser import MLP_BO_Optimiser
from math import sqrt
import numpy as np

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
        Custom Dataset class to handle CIFAR-10 from Pandas DataFrames.
        :param data: DataFrame containing features.
        :param labels: Series containing labels.
        :param transform: Optional transform to be applied to each image.
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




def run_mlp_classification(hidden1, hidden2, hidden3, hidden4, activation, lr, weight_decay):
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
    for epoch in range(3):  # Small number of epochs for optimization speed
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
for lengthscale_prior_mean in lengthscale_prior_means:
    
    averaged_y = 0
    best_y = 0
    for _ in range(3):
        result, best_accuracy, best_SVM_hyperparameters = optimiser.optimise(
            code_str = source_code_str,
            n_iter=30, initial_points=10, sample_per_batch=1,
            svm_kernel_lengthscale_prior_mean = sqrt(2),
            svm_kernel_outputscale_prior_mean = sqrt(2),
            svm_C_lengthscale_prior_mean = sqrt(2),
            svm_C_outputscale_prior_mean = lengthscale_prior_mean,
            svm_gamma_lengthscale_prior_mean = sqrt(2),
            svm_gamma_outputscale_prior_mean = sqrt(2),
            svm_coef0_lengthscale_prior_mean = sqrt(2),
            svm_coef0_outputscale_prior_mean = sqrt(2),
        )

        if best_accuracy >= best_y:
            best_y = best_accuracy
        averaged_y += best_y
    results_for_plotting[lengthscale_prior_mean] = averaged_y / 3

print(results_for_plotting)