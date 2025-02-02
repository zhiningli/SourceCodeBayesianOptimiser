# In this experiment, I am exploring if mistral AI is better at identifying the most relevant script despite noises such as 
import importlib
from src.mistral.mistral import MistralClient
import time

experiment4_prompts = """
Your task is to inspect the following target code and identify the most identical models from the following 10 candidates. Pay attention to noises such as comments, extra spacing and change of variable names that does not alter the functionality of models.

Here is the target codes: 
<target code>
{target_code}
</target code> 

Here are the candidate codes:
<candidate code>
<candidate_1>{candidate_code_1}</candidate1>
<candidate_2>{candidate_code_2}</candidate2>
<candidate_3>{candidate_code_3}</candidate3>
<candidate_4>{candidate_code_4}</candidate4>
<candidate_5>{candidate_code_5}</candidate5>
<candidate_6>{candidate_code_6}</candidate6>
<candidate_7>{candidate_code_7}</candidate7>
<candidate_8>{candidate_code_8}</candidate8>
<candidate_9>{candidate_code_9}</candidate9>
<candidate_10>{candidate_code_10}</candidate10>
</candidate code> 

You should just return the tag of the most relevant models. It must be one of the ten candidate codes.

Things to pay attention:
1. Commnets do not alter the functional requirements
2. Spaces do not alter the functional requirements
3. Change of variable names might affect the functional requirements. However, if all variables name are swapped. The functionality may not be changed.

Return the most similar model if there is no exact match

Example Output:
I have inspected all 10 candiate models (do you see 10 models?) The most simlar model is candidate XXX

"""
mistral = MistralClient()

# Initialize dictionaries for models and corrupted models
models = {}
corrupted_models = {}

for i in range(11, 21):
    module = importlib.import_module(f"src.scripts.models.model{i}")

    try:
        model_str = getattr(module, "model")
        corrupted_model_str = getattr(module, "corrupted_model")
    except AttributeError as e:
        print(f"Model {i}: Error loading model strings: {e}")
        continue

    if model_str is None or corrupted_model_str is None:
        print(f"Model {i}: Missing model or corrupted model string.")
        continue

    models[i] = model_str
    corrupted_models[i] = corrupted_model_str

for i in range(11, 21):
    print("Answer the most identical model to model", i)
    prompt = experiment4_prompts.format(
        target_code = models[i],
        candidate_code_1 = corrupted_models[11],
        candidate_code_2 = corrupted_models[12],
        candidate_code_3 = corrupted_models[13],
        candidate_code_4 = corrupted_models[14],
        candidate_code_5 = corrupted_models[15],
        candidate_code_6 = corrupted_models[16],
        candidate_code_7 = corrupted_models[17],
        candidate_code_8 = corrupted_models[18],
        candidate_code_9 = corrupted_models[19],
        candidate_code_10 = corrupted_models[20],        
    )

    mistral_response = mistral.call_codestral(prompt=prompt)

    print(mistral_response)

    time.sleep(10)

