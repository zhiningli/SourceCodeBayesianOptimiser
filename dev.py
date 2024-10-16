import numpy as np
from src.surrogate.kernels.RBF import RBF
from src.acquisition.EI import EI
from src.surrogate.GP import GP
from src.optimiser.optimiser import Optimiser

def black_box_function(x):
    return np.sin(5 * x) * (1 - np.tanh(x ** 2))


def dev_test():

    kernel = RBF(length_scale=0.5)
    surrogate = GP(kernel=kernel)
    acquisition_func = EI(xi=0.2)

    print("Surrogate model chosen: ", surrogate)

    optimiser = Optimiser(acquisition=acquisition_func, model=surrogate, n_iter = 10, objective_func=black_box_function)

    X_init = np.random.uniform(-2, 2, (5, 1))
    y_init = black_box_function(X_init)

    bounds = [(-2.0, 2.0)]

    optimiser.optimise(X_init, y_init, bounds)


if __name__ == '__main__':
    dev_test()