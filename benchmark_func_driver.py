from src.utils.benchmark_functions.benchmark_functions import Rastrigin, Beale, Sphere, BinaryTreeStructuredFunction


rastrigin = Rastrigin(n_dimension=2, noises=0.1, irrelevant_dims=0)

print(rastrigin.describe)
print(rastrigin.get_source_code())

