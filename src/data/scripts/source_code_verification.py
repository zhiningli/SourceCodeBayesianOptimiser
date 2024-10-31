import logging

from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.data.source_code_validator import SourceCodeValidator

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

###



exmaple_source_code = sourceCodeRepository.get_source_code(source_code_type="SVM")['source_code']
print(exmaple_source_code)

source_code_validator = SourceCodeValidator()

source_code_validator.iterative_refinement_and_validation(exmaple_source_code)

logging.info(source_code_validator.source_code)








