from src.scripts.full_script.scripts11 import code_str
from src.newIdeas.bo_optimiser import MLP_BO_Optimiser
import numpy as np

optimiser = MLP_BO_Optimiser()

search_space = {
    'learning_rate': np.logspace(-5, -1, num=50).tolist(),  # Logarithmically spaced values
    'momentum': [0.01 * x for x in range(100)],  # Linear space
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'num_epochs': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
}

accuracies, best_y, best_candidate = optimiser.optimise(code_str=code_str, search_space=search_space, objective_function_name="train_simple_nn", initial_points=10, n_iter=50)

#train_simple_nn(
#    learning_rate=0.02, 
#    momentum=0.9, 
#    batch_size=32, 
#    weight_decay=1e-3, 
#    num_epochs=50
#)

print("best_hyperparameter configuration: ", best_candidate)
print("best y so far: ", best_y)
print("accuracies: ", accuracies)
