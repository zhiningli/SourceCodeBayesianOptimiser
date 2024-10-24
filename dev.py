import numpy as np
import matplotlib.pyplot as plt
from src.surrogate.kernels.RBF import RBF
from src.acquisition.PI import PI
from src.surrogate.GP import GP
from src.optimiser.optimiser import Optimiser
from src.utils.benchmark_functions.benchmark_functions import Sphere, Rastrigin, Beale
import csv
import os

def dev_test():
    # Define the benchmark function to use for testing
    n_dimension = 2
    benchmark = Rastrigin(n_dimension=n_dimension)

    res = benchmark.evaluate([3.491966435,	-1.667119482])
    print("Testing benchmark function: ", res)

    print(f"Testing {benchmark.__class__.__name__} function with {n_dimension} dimensions")
    print(f"Search space: {benchmark.search_space}")
    print(f"Known global minimum: {benchmark.global_minimum}")
    print(f"Known global minimum location: {benchmark.global_minimumX}")

    # Set up the optimizer components
    kernel = RBF(length_scales=[0.25, 0.25])
    surrogate = GP(kernel=kernel, noise=1e-7)
    acquisition_func = PI(xi=0.1)
    optimiser = Optimiser(acquisition=acquisition_func, model=surrogate, n_iter=200, objective_func=benchmark.evaluate)

    # Define the bounds for optimization
    bounds = benchmark.search_space

    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    # Generate initial points, ensuring each dimension is sampled within its respective range
    X_init = np.random.uniform(lower_bounds, upper_bounds, (25, benchmark.n_dimension))
    y_init = np.array([benchmark.evaluate(x) for x in X_init])

    # Store the evaluation points for plotting
    evaluation_points = [X_init.copy()]

    # Run the optimization
    result = optimiser.optimise(X_init, y_init, bounds)

    # Collect all evaluation points
    evaluation_points.append(optimiser.model.X_train)
    print(evaluation_points)

    # Flatten the list of arrays into a single array for exporting
    all_points = np.vstack(evaluation_points)

    # Export the data to a CSV file
    export_data_to_csv(all_points, benchmark)

    # Display the result
    print("\nOptimisation result:")
    print(f"Best point found: {result['best_point']}")
    print(f"Best value found: {result['best_value']}")
    print(f"Known global minimum: {benchmark.global_minimum} at {benchmark.global_minimumX}")

    # Calculate the Euclidean distance between the found point and the known global minimum point
    distance_to_global_minimum = np.linalg.norm(result['best_point'] - benchmark.global_minimumX)
    print(f"Distance to global minimum: {distance_to_global_minimum}")

    # Check if the result is close to the known global minimum location
    if distance_to_global_minimum < 0.1:
        print("The optimiser found a point close to the global minimum in the input space!")
    else:
        print("The optimiser did not find a point close enough to the global minimum.")

def export_data_to_csv(points, benchmark):
    # Prepare data for export
    data = []
    for point in points:
        x, y = point
        value = benchmark.evaluate(point)
        print(x, y, value)
        data.append([x, y, value])
    
    # Get the absolute path of the current file's directory and construct the data folder path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    # Define the filename and full path
    filename = os.path.join(data_dir, 'optimization_data.csv')
    
    # Write to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'value'])  # Header
        writer.writerows(data)
    
    print(f"Data exported to {filename}")


if __name__ == '__main__':
    dev_test()
