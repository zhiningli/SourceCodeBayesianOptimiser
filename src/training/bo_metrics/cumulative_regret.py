import numpy as np


def cumulative_regret(iteration_values, true_optimal_value):
    """
    Calculate the cumulative regret for a Bayesian Optimization process.

    Parameters:
    iteration_values (list or numpy array): A list of objective function values for each iteration.
    true_optimal_value (float): The true optimal value of the objective function.

    Returns:
    float: The cumulative regret over all iterations.
    """
    iteration_values = np.array(iteration_values)
    regret_per_iteration = np.abs(true_optimal_value - iteration_values)
    cumulative_regret_value = np.sum(regret_per_iteration)
    return cumulative_regret_value

