import json
import numpy as np
from src.newIdeas.bo_optimiser import MLP_BO_Optimiser
from src.data.db.script_crud import ScriptRepository

# Load the JSON file containing bounds
with open("success_script.json", "r") as f:
    bounds_data = json.load(f)

# Define the initial search space
search_space = {
    'learning_rate': np.logspace(-5, -1, num=50).tolist(),  # Logarithmically spaced values
    'momentum': [0.01 * x for x in range(100)],  # Linear space
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'num_epochs': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
}

repo = ScriptRepository()
optimiser = MLP_BO_Optimiser()

for i in range(91, 101):
    # Load the lower and upper bounds for the current script
    bounds = bounds_data[str(i)]
    lower, upper = bounds[0], bounds[1]

    # Trim the search space based on bounds
    new_search_space = {
        'learning_rate': search_space['learning_rate'][int(lower[0]):int(upper[0]) + 1],
        'momentum': search_space['momentum'][int(lower[1]):int(upper[1]) + 1],
        'weight_decay': search_space['weight_decay'][int(lower[2]):int(upper[2]) + 1],
        'num_epochs': search_space['num_epochs'][int(lower[3]):int(upper[3]) + 1],
    }

    print("new_constrained_search_space: ", new_search_space)

    # Fetch the script from the database
    script_name = f"script{i}"
    target_script_object = repo.fetch_script_by_name(name=script_name)
    script_code_str = target_script_object["script"]

    # Run the optimization
    accuracies, best_y, best_candidate = optimiser.optimise(
        code_str=script_code_str,
        search_space=new_search_space,
        objective_function_name="train_simple_nn"
    )

    best_candidate = list(map(int, best_candidate.flatten().tolist()))

    # Update the script object with the optimization results
    target_script_object["constrained_search_space"] = {
        "search_space": {
            "lower": lower,
            "upper": upper,
        },
        "best_score": float(best_y),  # Convert best_y to a scalar
        "accuracies": list(map(float, accuracies)),  # Ensure accuracies is a list of floats
        "new_best_candidate": best_candidate,  # Convert best_candidate to a list of floats
        "best_hyperparameters": {
            "learning_rate": new_search_space["learning_rate"][int(best_candidate[0])],
            "momentum": new_search_space["momentum"][int(best_candidate[1])],
            "weight_decay": new_search_space["weight_decay"][int(best_candidate[2])],
            "num_epochs": new_search_space["num_epochs"][int(best_candidate[3])],
        },
    }

    # Update the script object in the database
    repo.update_script(script_object=target_script_object, script_name=script_name)
