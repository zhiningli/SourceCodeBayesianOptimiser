import numpy as np

def convergence_rate(iteration_values):
    """
    Calculate the convergence rate for a Bayesian Optimization process.

    Parameters:
    iteration_values (list or numpy array): A list of objective function values for each iteration.

    Returns:
    float: A measure of the convergence rate (smaller values indicate faster convergence).
    """
    if len(iteration_values) < 2:
        raise ValueError("At least two iterations are required to calculate the convergence rate.")
    
    # Convert to a numpy array for easier calculations
    iteration_values = np.array(iteration_values)
    
    # Track the best value found so far at each iteration
    best_so_far = np.minimum.accumulate(iteration_values)
    
    # Calculate the change in best value over iterations
    rate_of_change = np.diff(best_so_far)
    
    # Absolute convergence rate as the sum of negative changes (i.e., improvements)
    convergence_rate = -np.sum(rate_of_change[rate_of_change < 0])
    
    return convergence_rate
