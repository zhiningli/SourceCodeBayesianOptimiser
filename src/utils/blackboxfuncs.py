import numpy as np

def black_box_function(x):
    return np.sin(5 * x) * (1 - np.tanh(x ** 2)) 