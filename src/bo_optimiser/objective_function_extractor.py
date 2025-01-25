import logging

class Objective_function_extractor:

    def __init__(self):
        self.code_str = None
        self.objective_function = None
        self.namespace = {}

    def add_code_str(self, code_str):
        self.code_str = code_str
        self.run_code_str()

    def run_code_str(self):
        if not self.code_str:
            raise ValueError("No code string stored in the function extractor")
        try:
            exec(self.code_str, self.namespace)
            logging.INFO("Code string successfully executed")
        except:
            raise RuntimeError("Code string execution failed")
        
    def inspect_namespace(self):
        for function_name in self.namespace.keys():
            print(function_name)

    def get_function_from_code_string(self, function_name):
        if not self.code_str:
            raise ValueError("No code string stored in the function extractor")
        return self.namespace(function_name)

        


