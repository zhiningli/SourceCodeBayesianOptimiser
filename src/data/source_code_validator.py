import logging
import traceback
from src.data.mistral_prompts.data_validation_prompt import source_code_validation_prompt
from src.mistral.mistral import MistralClient

logging.basicConfig(level=logging.INFO)

class SourceCodeValidator:
    
    def __init__(self):
        self.mistral = MistralClient()
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
        bool: True if the code runs successfully
        """
        try:
            namespace = {}
            exec(code_str, namespace)
            namespace["run_svm_classification"]()  # Updated to match the function name in code_str
            return True
        except Exception as e:
            error_message = traceback.format_exc()
            logging.error("Execution failed with error: %s", error_message)
            self.last_error = error_message
            return False

    
    def refine_code_with_llm(self, code_str, error_msg):
        """
        Refine the source code using Claude to resolve errors.
        
        Parameters:
            code_str: the source code to refine
            error_msg: the error message from the last validation iteration
        
        Returns:
            str: Refined source code
        """
        prompt = source_code_validation_prompt.format(
            source_code=code_str,
            error_message=error_msg
        )
        
        refined_code = self.mistral.call_codestral(prompt)
        
        return refined_code
    
    
    def iterative_refinement_and_validation(self, initial_code, max_iteration=5):
        """
        Iteratively refine and validate the source code.
        """
        code = initial_code
        for i in range(max_iteration):
            logging.info(f"Iteration {i + 1}: Validating code...")
            if self.validate_source_code(code):
                self.source_code = code
                logging.info("Code validated and saved")
                return code
            else:
                response = self.refine_code_with_llm(code, self.last_error)
                code = self.mistral.extract_code_block(response)
        
        logging.warning("Max iteration reached, code could not be validated")
        return False
