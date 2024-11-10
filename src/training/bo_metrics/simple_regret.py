import numpy as np
def simple_regret(iteration_values, true_optimal_value):
    """
    Calculate the simple regret for a Bayesian Optimization process.

    Parameters:
    iteration_values (list or numpy array): A list of objective function values for each iteration.
    true_optimal_value (float): The true optimal value of the objective function.

    Returns:
    float: The simple regret.
    """
    iteration_values = np.array(iteration_values)
    best_found_value = np.min(iteration_values)  # Assuming minimization problem
    simple_regret_value = np.abs(true_optimal_value - best_found_value)
    return simple_regret_value