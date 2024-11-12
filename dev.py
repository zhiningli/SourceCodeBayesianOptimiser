import logging
from datetime import datetime
from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.preprocessing.SVM_BO_optimiser import SVM_BO_optimiser

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

# Step 1: Load source code with status of generated_from_template
source_code_status = SourceCodeStatus.FAILED_VALIDATION.value

source_code_pending_validation = sourceCodeRepository.get_source_code(name="SVM_source_code_on_sklearn_wine_dataset")

source_code = source_code_pending_validation["source_code"]
optimiser = SVM_BO_optimiser()


result = optimiser.optimise(source_code)

print(result)
