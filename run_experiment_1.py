from src.main_agregator import Constrained_Search_Space_Constructor
import importlib
from src.data.db.script_crud import ScriptRepository
from tqdm import tqdm
import json

script_repo = ScriptRepository()

def get_relevant_script_by_model_num_and_dataset_num(model_num, dataset_num):
    if model_num == 1:
        script_name = str(dataset_num)
    elif model_num in set([2, 3, 4, 5, 6, 7, 8, 9]):
        if dataset_num == 10:
            script_name = str(model_num)+"0"
        else:
            script_name = str(model_num-1)+str(dataset_num)
    elif model_num == 10:
        if dataset_num == 10:
            script_name = "100"
        else:
            script_name = "9" + str(dataset_num)
    return script_name  

def within_bound(actual_hyperparameter, lower, upper):

    for i in range(len(actual_hyperparameter)):
        if lower[i] > actual_hyperparameter[i] or upper[i] < actual_best_hyperparameters[i]:
            return False
    print("")
    return True 
                   

main_constructor = Constrained_Search_Space_Constructor()

success_count = 0
success_script = {}

for model_num in range(10, 11):
    for dataset_num in tqdm(range(1, 11)):
        script_name = get_relevant_script_by_model_num_and_dataset_num(model_num=model_num, dataset_num=dataset_num)

        try:
            module = importlib.import_module(f"src.scripts.full_script.scripts{script_name}")

            code_str = getattr(module, "code_str", None)

            lower, upper = main_constructor.suggest_search_space(
                code_str=code_str, target_model_num=model_num, target_dataset_num=dataset_num
            )

            script_object = script_repo.fetch_script_by_name("script" + script_name)
            actual_best_hyperparameters = script_object["best_candidate"]

            if within_bound(actual_best_hyperparameters, lower, upper):
                print("success!")
                success_count += 1
            success_script[script_name] = (lower, upper)
        except Exception as e:
            print(f"Error processing model {model_num}, dataset {dataset_num}: {e}")
            print("Returning success count early...")
            print("success_count: ", success_count)
            print("success_script:", success_script)


print("Final success_count: ", success_count)
print("success_script:", success_script )
with open("success_script_new.json", "w") as json_file:
    json.dump(success_script, json_file, indent=4)
    print("success_script saved to success_script.json")
