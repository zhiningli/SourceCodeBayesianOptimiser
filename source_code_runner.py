from importlib import import_module
from src.newIdeas.bo_optimiser import MLP_BO_Optimiser
import numpy as np
from src.data.db.script_crud import ScriptRepository

optimiser = MLP_BO_Optimiser()
search_space = {
    'learning_rate': np.logspace(-5, -1, num=50).tolist(),  # Logarithmically spaced values
    'momentum': [0.01 * x for x in range(100)],  # Linear space
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'num_epochs': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
}

repository = ScriptRepository()

for i in range(1, 101):
    # Dynamically construct the module name
    module_name = f"src.scripts.full_script.scripts{i}"
    
    # Import the module dynamically
    module = import_module(module_name)
    
    # Extract `code_str` from the imported module
    code_str = module.code_str

    # Perform optimization for each `code_str`
    accuracies, best_y, best_candidate = optimiser.optimise(
        code_str=code_str,
        search_space=search_space,
        objective_function_name="train_simple_nn",
        initial_points=25,
        n_iter=20
    )
    
    best_candidate = best_candidate.flatten()
    best_hyperparameter = {
        "learning_rate": search_space["learning_rate"][int(best_candidate[0].item())],
        "momentum": search_space["momentum"][int(best_candidate[1].item())],
        "weight_decay": search_space["weight_decay"][int(best_candidate[2].item())],
        "num_epochs": search_space["num_epochs"][int(best_candidate[3].item())]
    }

    script_object = {
        "script": code_str,
        "best_candidate": best_candidate.cpu().numpy().tolist(),  # Convert tensor to a list
        "best_hyperparameters": best_hyperparameter,  # Already a dictionary
        "best_score": float(best_y),  # 
        "accuracies": [float(acc) for acc in accuracies], 
    }

    print(script_object)
    repository.save_scripts(script_object)
