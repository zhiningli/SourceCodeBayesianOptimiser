import logging
from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.data.claude_prompts.data_validation_prompt import source_code_validation_prompt

logging.basicConfig(level=logging.INFO)

class SourceCodeValidator:

    def __init__(self, llm_refine_func):
        self.llm_refine_func = llm_refine_func
        self.source_code = None
        self.prompt = None
        self.last_error = None

    def set_source_code(self, source_code_object):
        self.source_code = source_code_object
    
    def validate_source_code(self, code_str):
        """
        Validate the provided source code by executing it

        Parameters:
        code_str

        Return:
        bool: True if the code is run successfully
        """
        try:
            namespace = {}
            exec(code_str, namespace)
            namespace["run_classification"]()
            return True
        except Exception as e:
            logging.error("Execution failed with error: %s", e)
            self.last_error = str(e)
            return False

    def refine_code_with_llm(self, code_str, error_msg):
        """
        Refine the source code using an LLM to resolve errors.
        
        Parameters:
            code_str: the source code to refine
            error_msg: the error message from the last validation iteration
        
        Returns:
            str: Refined source code
        """

        prompt = f"""
        Refactor the following code to resolve the error: "{error_msg}"
        Code:
        {code_str}
        """

        refined_code = self.llm_refine_func(prompt)
        return refined_code
    
    def iterative_refinement_and_validation(self, initial_code, max_iteration = 5):
        """
        Iteratively refine and validates the source code
        """
        code = initial_code
        for i in range(max_iteration):
            logging.info(f"Iteration {i+1}: Validating code...")
            if self.validate_source_code(code):
                self.source_code = code
                logging.info("Code validated and saved")
                return True
            else:
                code = self.refine_code_with_llm(code, self.last_error)
        
        logging.warning("Max iteration reached, code could not be validated")
        return False