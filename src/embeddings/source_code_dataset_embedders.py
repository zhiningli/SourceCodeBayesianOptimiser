import numpy as np
import itertools
from tqdm import tqdm

class Dataset_Scoring_Helper:

    def __init__(self):

        self.search_space = {
                                'learning_rate': np.logspace(-5, -1, num=50).tolist(),
                                'momentum': [0.01 * x for x in range(100)],
                                'weight_decay': [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                                'num_epochs': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
                            }
        self.initial_points = [
            [22,  1,  1,  9], [45, 68,  4,  4], [36, 46,  0, 14],
            [ 6, 84,  3,  7], [ 1, 36,  4,  6], [29, 95,  2, 15],
            [39, 17,  3,  0], [16, 51,  1, 12], [13, 42,  3, 16],
            [42, 75,  1,  4], [26, 10,  4, 12], [ 4, 70,  2,  2],
            [ 9, 21,  0,  2], [34, 59,  4, 10], [48, 25,  2,  8],
            [19, 92,  5, 14], [21, 28,  1,  1], [47, 90,  3, 13],
            [31, 25,  2,  6], [12, 56,  4, 15]
        ]
        self.objective_func = None

    def load_objective_function(self, code_str, objective_function_name):
        namespace = {}
        exec(code_str, namespace)
        self.objective_func = namespace.get(objective_function_name)
    
    def execute_objective_func_against_inital_points(self):
        
        results = {}
        for i, initial_point in enumerate(tqdm(self.initial_points, desc="Processing initial points")):
            params = {
                "learning_rate": self.search_space["learning_rate"][initial_point[0]],
                "momentum": self.search_space["momentum"][initial_point[1]],
                "weight_decay": self.search_space["weight_decay"][initial_point[2]],
                "num_epochs": self.search_space["num_epochs"][initial_point[3]]
            }

            results[str(i)] = {
                "hyperparameters": params,
                "value":self.objective_func(**params)}

        return results

    def kendall_tau_rank_correlation(self, d1_scores, d2_scores):

        num_configs = len(d1_scores)

        d1_scores = np.array(d1_scores)
        d2_scores = np.array(d2_scores)

        total_pairs = num_configs * (num_configs - 1) // 2

        concordant_discordant_count = 0

        for (i, j) in itertools.combinations(range(num_configs), 2):
            d1_relation = np.sign(d1_scores[i] - d1_scores[j])
            d2_relation = np.sign(d2_scores[i] - d2_scores[j])

            if d1_relation == d2_relation:
                concordant_discordant_count += 1
            else:
                concordant_discordant_count -= 1

        ktrc = concordant_discordant_count / total_pairs

        return ktrc


        


    