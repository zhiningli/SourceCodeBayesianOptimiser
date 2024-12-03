from src.preprocessing.MLP_BO_optimiser import MLP_BO_Optimiser
import random
import torch
import numpy as np

code_str = """
import openml
import numpy as np
import pandas as pd
import torch
import random
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


def load_openml_cifar10(seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Load dataset as a DataFrame
    dataset = openml.datasets.get_dataset(dataset_id=40927)  # CIFAR-10 dataset ID

    # Get data and ensure target is included
    df, _, _, _ = dataset.get_data(dataset_format="dataframe")

    # Shuffle the dataset and select only 6000 samples
    df = df.sample(n=6000, random_state=seed).reset_index(drop=True)

    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)  # Exclude target column
    y = df.iloc[:, -1].to_numpy(dtype=np.int64)  # Extract target column

    # Wrap features and labels into pandas DataFrame/Series
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y, name="label")

    # Split dataset into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(X_df))  # 4800 samples for training
    X_train = X_df.iloc[:train_size].reset_index(drop=True)
    y_train = y_series.iloc[:train_size].reset_index(drop=True)
    X_val = X_df.iloc[train_size:].reset_index(drop=True)  # 1200 samples for validation
    y_val = y_series.iloc[train_size:].reset_index(drop=True)

    return X_train, y_train, X_val, y_val


def run_mlp_classification(conv_feature_num, hidden1, conv_kernel_size, conv_stride, activation, lr, weight_decay, epoch, batch_size):
    # Load CIFAR-10 dataset from OpenML
    X_train, y_train, X_val, y_val = load_openml_cifar10()

    # Define transformations
    def transform(image):
        return preprocess(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # Create PyTorch datasets
    train_dataset = CIFAR10Dataset(X_train, y_train, transform=transform)
    val_dataset = CIFAR10Dataset(X_val, y_val, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model definition
    activation_fn = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}[activation]
    model = nn.Sequential(
        nn.Conv2d(3, conv_feature_num, kernel_size=conv_kernel_size, stride=conv_stride, padding=int(conv_kernel_size) // 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(int(conv_feature_num * 32 ** 2 / conv_stride ** 2), hidden1),
        nn.Dropout(p=0.4),
        activation_fn(),
        nn.Linear(hidden1, 10),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for ep in range(epoch):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            # Track training metrics
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

        # Compute training loss and accuracy
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Validation loop
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(x_batch)
                _, predicted = torch.max(output, 1)
                total_val += y_batch.size(0)
                correct_val += (predicted == y_batch).sum().item()

        # Compute validation accuracy
        val_accuracy = correct_val / total_val

    # Return final validation accuracy
    return val_accuracy


"""

search_space = {
    "MLP_conv_feature_num_nu": [0.5, 1.5, 2.5],
    "MLP_conv_kernel_size_nu": [0.5, 1.5, 2.5],
    "MLP_conv_stride_nu": [0.5, 1.5, 2.5],
    "MLP_hidden1_nu": [0.5, 1.5, 2.5],
    "MLP_lr_nu": [0.5, 1.5, 2.5],
    "MLP_activation_nu": [0.5, 1.5, 2.5],
    "MLP_weight_decay_nu": [0.5, 1.5, 2.5],
    "MLP_epoch_nu": [0.5, 1.5, 2.5],
    "MLP_batch_size_nu": [0.5, 1.5, 2.5],
}

results_for_plotting = {
    'descrpt': "BO results after selecting different nu for matern kernel in different input dimensions",
    'experiments': []
}

# Run experiments
for _ in range(5):
    random_idx = [random.randint(0, 2) for _ in range(9)]
    print(random_idx)

    # Get the hyperparameter values
    random_values = [
        search_space["MLP_conv_feature_num_nu"][random_idx[0]],
        search_space["MLP_conv_kernel_size_nu"][random_idx[1]],
        search_space["MLP_conv_stride_nu"][random_idx[2]],
        search_space["MLP_hidden1_nu"][random_idx[3]],
        search_space["MLP_lr_nu"][random_idx[4]],
        search_space["MLP_activation_nu"][random_idx[5]],
        search_space["MLP_weight_decay_nu"][random_idx[6]],
        search_space["MLP_epoch_nu"][random_idx[7]],
        search_space["MLP_batch_size_nu"][random_idx[8]],
    ]

    optimiser = MLP_BO_Optimiser()

    experiment_results = []
    for i in range(3):  # Run 3 experiments for the same configuration
        accuracies, best_y, best_candidate = optimiser.optimise(
            code_str=code_str,
            MLP_conv_feature_num_nu=random_values[0],
            MLP_conv_kernel_size_nu=random_values[1],
            MLP_conv_stride_nu=random_values[2],
            MLP_hidden1_nu=random_values[3],
            MLP_lr_nu=random_values[4],
            MLP_activation_nu=random_values[5],
            MLP_weight_decay_nu=random_values[6],
            MLP_epoch_nu=random_values[7],
            MLP_batch_size_nu=random_values[8],
        )

        print(f"Run {i + 1}: accuracies={accuracies}, best_y={best_y}, best_candidate={best_candidate}")

        experiment_results.append({
            'run': i + 1,
            'accuracies': accuracies,
            'best_y': best_y,
            'best_candidate': best_candidate
        })

    # Store results for this hyperparameter configuration
    results_for_plotting['experiments'].append({
        'hyperparameters': random_values,
        'random_idx': random_idx,
        'results': experiment_results
    })

# Save results to a file
output_file = "bo_results.txt"
with open(output_file, "w") as f:
    f.write(repr(results_for_plotting))

print(f"Results saved to {output_file}")