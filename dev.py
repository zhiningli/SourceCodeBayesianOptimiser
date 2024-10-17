import numpy as np
import matplotlib.pyplot as plt
from src.surrogate.kernels.RBF import RBF
from src.acquisition.PI import PI
from src.surrogate.GP import GP
from src.optimiser.optimiser import Optimiser
from src.utils.benchmark_functions import Sphere
from mpl_toolkits.mplot3d import Axes3D

def dev_test():
    # Define the benchmark function to use for testing
    n_dimension = 2
    benchmark = Sphere(n_dimension=n_dimension)

    print(f"Testing {benchmark.__class__.__name__} function with {n_dimension} dimensions")
    print(f"Search space: {benchmark.search_space}")
    print(f"Known global minimum: {benchmark.global_minimum}")
    print(f"Known global minimum location: {benchmark.global_minimumX}")

    # Set up the optimizer components
    kernel = RBF(length_scale=1)
    surrogate = GP(kernel=kernel, noise=1e-2)
    acquisition_func = PI(xi=0.01)
    optimiser = Optimiser(acquisition=acquisition_func, model=surrogate, n_iter=100, objective_func=benchmark.evaluate)

    # Generate initial points within the search space
    X_init = np.random.uniform(-5.12, 5.12, (15, 2))  # 15 initial points in a 2D space
    y_init = np.array([benchmark.evaluate(x) for x in X_init])

    # Define the bounds for optimization
    bounds = benchmark.search_space

    print("\nInitial points and their evaluations:")
    for x, y in zip(X_init, y_init):
        print(f"Point: {x}, Value: {y}")

    # Store the evaluation points for plotting
    evaluation_points = [X_init.copy()]

    # Run the optimization
    result = optimiser.optimise(X_init, y_init, bounds)

    # Collect all evaluation points
    evaluation_points.append(optimiser.model.X_train)

    # Flatten the list of arrays into a single array for plotting
    all_points = np.vstack(evaluation_points)

    # Display the result
    print("\nOptimisation result:")
    print(f"Best point found: {result['best_point']}")
    print(f"Best value found: {result['best_value']}")
    print(f"Known global minimum: {benchmark.global_minimum} at {benchmark.global_minimumX}")

    # Check if the result is close to the known global minimum
    if np.isclose(result['best_value'], benchmark.global_minimum, atol=1e-2):
        print("The optimiser found a value close to the global minimum!")
    else:
        print("The optimiser did not converge to the expected global minimum.")

    # Plot the Sphere function's contour and evaluation points
    plot_sphere_function_3d(benchmark, all_points)

def plot_sphere_function_3d(benchmark, evaluation_points):
    # Define the grid for plotting
    x = np.linspace(-5.12, 5.12, 400)
    y = np.linspace(-5.12, 5.12, 400)
    X, Y = np.meshgrid(x, y)
    Z = benchmark.evaluate([X, Y])

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.7, cmap='viridis', edgecolor='none')

    # Plot the evaluation points
    ax.scatter(evaluation_points[:, 0], evaluation_points[:, 1], 
               [benchmark.evaluate(point) for point in evaluation_points], 
               color='red', s=50, label='Evaluation Points', alpha=0.7)
    ax.scatter(0, 0, 0, color='white', marker='x', s=100, label='Global Minimum (0, 0)')

    # Labels and title
    ax.set_title('3D Sphere Function with Evaluation Points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Function Value')
    ax.view_init(elev=30, azim=-60)  # Adjust the viewing angle for better perspective
    ax.legend()

    # Save the plot as an image file
    plt.savefig('sphere_function_3d_optimization.png')
    print("3D plot saved as 'sphere_function_3d_optimization.png'")

if __name__ == '__main__':
    dev_test()
