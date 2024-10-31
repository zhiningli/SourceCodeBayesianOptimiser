import logging
from src.data.claude_prompts.data_validation_prompt import source_code_validation_prompt
from src.claude.claude import Claude 

logging.basicConfig(level=logging.INFO)

class SourceCodeValidator:
    
    def __init__(self):
        self.claude = Claude()
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
        
        refined_code = self.claude.call_claude(prompt)
        
        return refined_code.strip()
    
    
    def iterative_refinement_and_validation(self, initial_code, max_iteration=5):
        """
        Iteratively refine and validate the source code.
        """
        code = initial_code
        logging.info(code)
        for i in range(max_iteration):
            logging.info(f"Iteration {i + 1}: Validating code...")
            if self.validate_source_code(code):
                self.source_code = code
                logging.info("Code validated and saved")
                return True
            else:
                scratchpad = self.refine_code_with_llm(code, self.last_error)
                code = self.claude.extract_tag(scratchpad, "source_code")
        
        logging.warning("Max iteration reached, code could not be validated")
        return False
