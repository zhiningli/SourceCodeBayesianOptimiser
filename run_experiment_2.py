'''
The main purpose of this experiment is to verify that
1. The algorithm proposed in experiment 1 can work when the code str and database is not seen by the past problems
'''
from src.main_agregator import Constrained_Search_Space_Constructor
import importlib
from src.data.db.script_crud import ScriptRepository
from src.bo_optimiser.bo_optimiser import MLP_BO_Optimiser
from src.middleware import ComponentStore
import numpy as np

script_repo = ScriptRepository()

unseen_model1 = """
# I am just adding some random comments to disrupt the vector embeddings and this by right should give some problems to the implementations.
import torch.nn.functional as F

class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_features, bottleneck_features=16):
        super(BottleneckResidualBlock, self).__init__()
        # First linear layer
        self.fc1 = nn.Linear(in_features, bottleneck_features)
        # Second with bottleneck features 
        self.fc2 = nn.Linear(bottleneck_features, bottleneck_features) 
        self.fc3 = nn.Linear(bottleneck_features, in_features) 
        self.relu = nn.ReLU()

        # The actuall passing functions
    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x += residual  # Skip connection
        return F.relu(x)  # Apply activation to the output

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.res1 = BottleneckResidualBlock(256, 64)
        self.res2 = BottleneckResidualBlock(256, 64)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.fc2(x)
        return x
"""


def get_relevant_script_by_model_num_and_dataset_num(model_num, dataset_num):
    if model_num == 1:
        script_name = str(dataset_num)
    elif model_num in set([2, 3, 4, 5, 6, 7, 8, 9]):
        if dataset_num == 10:
            script_name = str(model_num)+"0"
        else:
            script_name = str(model_num-1)+str(dataset_num)
    elif model_num == 10:
        if dataset_num == 10:
            script_name = "100"
        else:
            script_name = "9" + str(dataset_num)
    return script_name  
                   
store = ComponentStore()


main_constructor = Constrained_Search_Space_Constructor()
main_constructor.store = store


dataset_num = 3

module = importlib.import_module(f"src.scripts.datasets.dataset{dataset_num}")
dataset = getattr(module, "dataset", None)

new_script = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification

""" + dataset + unseen_model1 + """
def train_simple_nn(learning_rate, momentum, weight_decay, num_epochs):
    # Initialize model, loss function, and optimizer
    model = Model(input_size=25, num_classes=3)  # Adjust input size to match your dataset
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
    # Testing the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
"""



lower, upper = main_constructor.suggest_search_space(
    code_str=new_script, target_model_num=None, target_dataset_num=dataset_num
)


# Define the initial search space
search_space = {
    'learning_rate': np.logspace(-5, -1, num=50).tolist(),  # Logarithmically spaced values
    'momentum': [0.01 * x for x in range(100)],  # Linear space
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'num_epochs': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
}

new_search_space = {
    'learning_rate': search_space['learning_rate'][int(lower[0]):int(upper[0]) + 1],
    'momentum': search_space['momentum'][int(lower[1]):int(upper[1]) + 1],
    'weight_decay': search_space['weight_decay'][int(lower[2]):int(upper[2]) + 1],
    'num_epochs': search_space['num_epochs'][int(lower[3]):int(upper[3]) + 1],
}


repo = ScriptRepository(collection_name="experiment2")
optimiser = MLP_BO_Optimiser()
optimiser.store = store

optimiser.objective_func = new_script

accuracies, best_y, best_candidate = optimiser.optimise(search_space=search_space)

constrained_accuracies, constrained_best_y, constrained_best_candidate = optimiser.optimise(search_space = new_search_space)

best_candidate = list(map(int, best_candidate.flatten().tolist()))
new_best_candidate = list(map(int, constrained_best_candidate.flatten().tolist()))

script_object_to_store = {
    "script": new_script,
    "best_candidate": best_candidate,
    "best_hyperparameters": {
        "learning_rate": search_space["learning_rate"][int(best_candidate[0])],
        "momentum": search_space["momentum"][int(best_candidate[1])],
        "weight_decay": search_space["weight_decay"][int(best_candidate[2])],
        "num_epochs": search_space["num_epochs"][int(best_candidate[3])],
    },
    "best_score": float(best_y),
    "accuracies": list(map(float, accuracies)),
    "scriptName": "unfamiliar script 3",
    "constrained_search_space": {
        "search_space": {
            "lower": lower,
            "upper": upper,
        },
        "best_score": float(constrained_best_y),
        "accuracies": list(map(float, constrained_accuracies)),
        "new_best_candidate": new_best_candidate,
        "best_hyperparameters": {
        "learning_rate": new_search_space["learning_rate"][int(new_best_candidate[0])],
        "momentum": new_search_space["momentum"][int(new_best_candidate[1])],
        "weight_decay": new_search_space["weight_decay"][int(new_best_candidate[2])],
        "num_epochs": new_search_space["num_epochs"][int(new_best_candidate[3])],
        },
    }    
}

repo.save_scripts(script_object_to_store)







