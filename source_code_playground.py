from src.preprocessing.SVM_BO_optimiser import SVM_BO_optimiser
from math import sqrt
import numpy as np

source_code_str = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Define Hyperparameters
hyperparameters = {
    'epochs': 20,
    'learning_rate': 0.001,
    'batch_size': 64,
    'momentum': 0.9,
    'dropout_rate': 0.5,
    'conv1_filters': 32,
    'conv2_filters': 64,
    'fc_neurons': 256,
    'l2_reg_conv1': 0.0001,
    'l2_reg_conv2': 0.0001,
    'l2_reg_fc': 0.0001
}

# 2. Prepare CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

# 3. Define CNN Model
class CNN(nn.Module):
    def __init__(self, hp):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, hp['conv1_filters'], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hp['conv1_filters'], hp['conv2_filters'], kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(hp['conv2_filters'] * 8 * 8, hp['fc_neurons'])
        self.fc2 = nn.Linear(hp['fc_neurons'], 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(hp['dropout_rate'])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN(hyperparameters)
print(model)

# 4. Define Loss Function with Regularization
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=hyperparameters['learning_rate'],
    momentum=hyperparameters['momentum'],
    weight_decay=(
        hyperparameters['l2_reg_conv1'] + 
        hyperparameters['l2_reg_conv2'] + 
        hyperparameters['l2_reg_fc']
    )  # L2 Regularization
)

# 5. Train the Model
def train_model(model, train_loader, criterion, optimizer, hp):
    model.train()
    train_losses = []

    for epoch in range(hp['epochs']):
        epoch_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{hp['epochs']}], Loss: {avg_loss:.4f}")
    
    return train_losses

train_losses = train_model(model, train_loader, criterion, optimizer, hyperparameters)

# 6. Evaluate the Model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy

accuracy = evaluate_model(model, test_loader)

# 7. Plot Training Loss
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

"""

optimiser = SVM_BO_optimiser()

lengthscale_prior_means = np.linspace(1, 20, 40)

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