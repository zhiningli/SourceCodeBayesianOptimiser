from mistralai import Mistral
import re
import logging

class MistralClient:

    def __init__(self,
                 api_key: str = "YT6c7Dvr5UTLNpNskkSG66bgUriDlfcZ"):
        self.model = "codestral-mamba-latest"
        self.client = Mistral(api_key=api_key)

    def call_codestral(self,
                    prompt: str,
                    ):
        
        chat_response = self.client.chat.complete(
            model = self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        
        response = chat_response.choices[0].message.content
        return response    
    
    def extract_code_block(self, text: str):
        # Regex pattern to capture code content, optionally ignoring the language identifier
        structure = r"```(?:\w+)?\n(.*?)```"
        match = re.search(structure, text, re.DOTALL)
        if match:
            code_content = match.group(1).strip()
            return code_content
        else:
            logging.warning("No code block found enclosed by triple backticks.")
            raise ValueError("No code block enclosed by triple backticks was found in the text")

