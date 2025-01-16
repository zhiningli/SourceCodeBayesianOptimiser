from src.embeddings.source_code_parser import Source_Code_Parser
from src.embeddings.source_code_embedders import Codebert_Embedder
from src.embeddings.source_code_dataset_embedders import Dataset_Scoring_Helper
import torch.nn.functional as F
from src.data.db.model_crud import ModelRepository
from src.data.db.script_crud import ScriptRepository
import torch
import heapq

class Constrained_Search_Space_Constructor:

    def __init__(self):

        self.parser = Source_Code_Parser()
        self.model_embedder = Codebert_Embedder()
        self.dataset_embedder = Dataset_Scoring_Helper()
        self.model_repository = ModelRepository()
        self.script_repository = ScriptRepository()

        self.model_code_str = None
        self.dataset_code_str = None
        self.overall_code_str = None

    def suggest_search_space(self, code_str, target_model_num, target_dataset_num):

        self.overall_code_str = code_str
        print("Step 1: extract relevant information from code_str...")
        information = self.parser.extract_information_from_code_string(code_str=code_str)
        print("Step 1 completed")
        print("Step 2: parse model and dataset code string...")
        self.model_code_str = information["model"]
        self.dataset_code_str = information["dataset"]
        print("Step 2 Completed")
        print("Step 3: computing model similarities...")
        model_similarities = self.compute_top_k_model_similarities()
        print("Step 3 completed")
        print("Step 4: compute dataset similarities...")
        dataset_similarities = self.compute_top_k_dataset_similarities(model_num=target_model_num)

        print("top_3_model_similarities: ", model_similarities)
        print("top_3_dataset_similarities: ", dataset_similarities) 
        print("Step 4 completed")
        print("Step 5: Constructing a constrained search space: ")

        hyperparameter_space = []

        for _, model_name in model_similarities:
            for _, dataset_name in dataset_similarities:
                model_num = int(model_name[5:])
                dataset_num = int(dataset_name[7:])
                if model_num == target_model_num and dataset_num == target_dataset_num:
                    continue
                relevant_script_object = self._get_relevant_script_by_model_num_and_dataset_num(model_num=model_num, dataset_num=dataset_num)
                best_candidate = relevant_script_object["best_candidate"]
                hyperparameter_space.append(best_candidate)
        print("hyperparameter_space: ", hyperparameter_space)
        search_space = self._construct_compact_hyperparameter_search_space(hyperparameter_space)

        print("constrained_search_space_constructed: ", search_space)
        return search_space

    def compute_top_k_model_similarities(self, k = 3):

        model_objects = self.model_repository.fetch_all_models()
        similarities = []
        target_embeddings = self.model_embedder.embed_source_code(self.model_code_str).squeeze()

        similarities = []
        for model_object in model_objects:
            model_name = model_object["model_name"]
            model_embeddings = torch.Tensor(model_object["model_embeddings"]).squeeze()
            cosine_sim = F.cosine_similarity(target_embeddings, model_embeddings, dim=0)
            heapq.heappush(similarities, (-cosine_sim.item(), model_name))  

        items = [heapq.heappop(similarities) for _ in range(len(similarities))]

        raw_cosines = [-item[0] for item in items] 
        min_sim, max_sim = min(raw_cosines), max(raw_cosines)

        normalized_items = [
            ((cosine - min_sim) / (max_sim - min_sim), name) for cosine, name in zip(raw_cosines, [item[1] for item in items])
        ]
        print("normalised_ranked_model_similarities: ", normalized_items)

        for norm_sim, name in normalized_items:
            heapq.heappush(similarities, (-norm_sim, name))  # Negate for max-heap

        # Extract the top 3 largest values from the heap
        res = []
        for _ in range(min(k, len(similarities))):
            normalized_sim, name = heapq.heappop(similarities)
            res.append((-normalized_sim, name))
        return res

    def compute_top_k_dataset_similarities(self, model_num, k=3):
        self.dataset_embedder.load_objective_function(self.overall_code_str, "train_simple_nn")

        # Evaluate the target dataset and extract metrics
        target_dataset_evaluation_results = self.dataset_embedder.execute_objective_func_against_inital_points()
        target_dataset_rank = self.extract_evaluation_metrics(target_dataset_evaluation_results)
        
        source_scripts_name_to_fetch = self._get_relevant_script_by_model_num(model_num=model_num)
        print("fetching following script from the database: ", source_scripts_name_to_fetch)
        
        dataset_similarities = []

        for source_script_name in source_scripts_name_to_fetch:
            script_object = self.script_repository.fetch_script_by_name(source_script_name)
            pre_trained_dataset_results = script_object["dataset_results"]
            pre_trained_dataset_rank = self.extract_evaluation_metrics(pre_trained_dataset_results)
            ktrc = self.dataset_embedder.kendall_tau_rank_correlation(target_dataset_rank, pre_trained_dataset_rank)

            # Add to the heap
            heapq.heappush(dataset_similarities, (-ktrc, script_object["script_name"]))

        # Extract all items from the heap for normalization
        items = [heapq.heappop(dataset_similarities) for _ in range(len(dataset_similarities))]

        # Extract the raw ktrc values and calculate min/max
        raw_ktrc = [-item[0] for item in items]  # Original ktrc values
        min_ktrc, max_ktrc = min(raw_ktrc), max(raw_ktrc)

        # Normalize ktrc values to 0-1
        normalized_items = [
            ((ktrc - min_ktrc) / (max_ktrc - min_ktrc) if max_ktrc != min_ktrc else 0, name)
            for ktrc, name in zip(raw_ktrc, [item[1] for item in items])
        ]
        print("normalised_ranked_dataset_similarities: ", normalized_items)
        # Push normalized values back into the heap
        for norm_ktrc, name in normalized_items:
            num = int(name[6:])
            if num == 10:
                heapq.heappush(dataset_similarities, (-norm_ktrc, "dataset10"))
            else:
                num %= 10
                heapq.heappush(dataset_similarities, (-norm_ktrc, "dataset"+str(num)))
        # Extract the top-k largest values from the normalized heap
        res = []
        for _ in range(min(k, len(dataset_similarities))):
            normalized_ktrc, name = heapq.heappop(dataset_similarities)
            res.append((-normalized_ktrc, name))  # Convert back to positive value

        return res


    def extract_evaluation_metrics(self, evaluation_results):
        """
        Extracts the 'value' field from the evaluation results.

        Args:
            evaluation_results (dict): A dictionary of evaluation results.

        Returns:
            list: A list of 'value' fields from the evaluation results.
        """
        return [data['value'] for data in evaluation_results.values()]
    
    def _get_relevant_script_by_model_num(self, model_num):
        if model_num == 1:
            return ["script" + str(i) for i in range(1, 10)] + ["script" + str(10)]
        elif model_num in set([2, 3, 4, 5, 6, 7, 8, 9]):
            return ["script" + str(model_num-1) + str(i) for i in range(1, 10)] + ["script" + str(model_num)+"0"]
        else:
            return ["script" + str(model_num-1) + str(i) for i in range(1, 10)] + ["script100"]

    def _get_relevant_script_by_model_num_and_dataset_num(self, model_num, dataset_num):
        if model_num == 1:
            script_name = "script"+str(dataset_num)
        elif model_num in set([2, 3, 4, 5, 6, 7, 8, 9]):
            if dataset_num == 10:
                script_name = "script"+str(model_num)+"0"
            else:
                script_name = "script"+str(model_num-1)+str(dataset_num)
        elif model_num == 10:
            if dataset_num == 10:
                script_name = "script100"
            else:
                script_name = "script9"+str(dataset_num)
        else:
            raise ValueError("Model number is wrong: ", model_num)
        print("fetching", script_name, "that contains model", model_num, "dataset_num", dataset_num)
        return self.script_repository.fetch_script_by_name(script_name)     
    
    def _construct_compact_hyperparameter_search_space(self, hyperparameter_space):
        l = [float('inf'), float('inf'), float('inf'), float('inf')]
        h = [0, 0, 0, 0]                
        for bound in hyperparameter_space:
            for i in range(len(bound)):
                l[i] = min(l[i], bound[i])
                h[i] = max(h[i], bound[i])

        return l, h


        





        
    


    