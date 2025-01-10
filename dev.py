from src.data.db.script_crud import ScriptRepository
from src.embeddings.source_code_parser import Source_Code_Parser
from src.embeddings.source_code_embedders import Codebert_Embedder
from tqdm import tqdm  # Import tqdm for progress bar
from src.embeddings.source_code_dataset_embedders import Dataset_Scoring_Helper

# Initialize the repository and dependencies
repo = ScriptRepository()
scripts = repo.fetch_all_scripts()
embedder = Codebert_Embedder()
parser = Source_Code_Parser()
dataset_scoring_helper = Dataset_Scoring_Helper()

# Wrap the scripts in tqdm for progress tracking
for i, script_object in enumerate(tqdm(scripts, desc="Processing scripts")):
    script_name = script_object["script_name"]
    code_str = script_object["script"]
    dataset_scoring_helper.load_objective_function(code_str, "train_simple_nn")
    results = dataset_scoring_helper.execute_objective_func_against_inital_points()
    print(results)
    script_object["dataset_results"] = results

    # Update the script in the database
    repo.update_script(script_object, script_name)
