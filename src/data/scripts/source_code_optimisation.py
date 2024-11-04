import logging
from datetime import datetime
from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.data.source_code_optimiser import SVMSourceCodeOptimiser

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

# Step 1: Load source code with status of generated_from_template
source_code_status = SourceCodeStatus.VALIDATED_TO_RUN.value

source_codes_pending_optimisation = sourceCodeRepository.get_source_code(source_code_type="SVM")["source_code"]

print(source_codes_pending_optimisation)


source_code_optimiser = SVMSourceCodeOptimiser()

source_code_optimiser.optimise(source_codes_pending_optimisation)