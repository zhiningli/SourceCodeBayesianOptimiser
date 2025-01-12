from src.main_agregator import Constrained_Search_Space_Constructor
from src.scripts.full_script.scripts1 import code_str

main_constructor = Constrained_Search_Space_Constructor()
print("Main constructor initiated successfully")

main_constructor.suggest_search_space(code_str=code_str, model_num=1)
