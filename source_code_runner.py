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




def run_mlp_classification(hidden, conv_kernel_size, conv_stride, activation, max_pool_kernel_size, lr, weight_decay, max_pool_stride, epoch, batch_size):
    # Load CIFAR-10 dataset from OpenML
    X_train, y_train, X_val, y_val = load_openml_cifar10()

    # Define transformations
    def transform(image):
        return preprocess(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # Create PyTorch datasets
    train_dataset = CIFAR10Dataset(X_train, y_train, transform=transform)
    val_dataset = CIFAR10Dataset(X_val, y_val, transform=transform)

    # DataLoaders
    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model definition
    input_size = 3 * 32 * 32  # CIFAR-10 images are 3x32x32
    activation_fn = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}[activation]
    model = nn.Sequential(
        # Convolutional Layer
        nn.Conv2d(3, 32, kernel_size=conv_kernel_size, stride = conv_stride, padding=int(conv_kernel_size)//2),
        nn.ReLU(),  
        nn.MaxPool2d(kernel_size=max_pool_kernel_size, stride= max_pool_stride, padding=int(max_pool_kernel_size)//2),

        nn.Flatten(), 
        nn.Linear(int(32**3 / (conv_stride**2 * max_pool_stride**2)), hidden),  
        nn.ReLU(),
        nn.Linear(hidden, 10) 
)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epoch):  # Small number of epochs for optimization speed
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        # Compute average training loss and accuracy for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(x_batch)
                loss = criterion(output, y_batch)

                # Accumulate validation loss
                val_loss += loss.item()

                # Calculate validation accuracy
                _, predicted = torch.max(output, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # Print metrics for the epoch
        print(f"Epoch [{epoch + 1}/{epoch}], "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, "
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")

        # Set model back to training mode
        model.train()

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


accuracy = run_mlp_classification(
    conv_kernel_size = 7,
    conv_stride = 1,
    max_pool_kernel_size = 3, 
    max_pool_stride = 2,
    hidden=1024,
    activation="ReLU",
    lr=0.01,
    weight_decay=0.01,
    epoch = 25,
    batch_size=64
)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
