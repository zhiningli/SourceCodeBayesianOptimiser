import logging
from datetime import datetime
from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.data.source_code_optimiser import SVMSourceCodeOptimiser

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

# Step 1: Load source code with status of generated_from_template
source_code_status = SourceCodeStatus.ABLE_TO_RUN_BO.value

source_codes_pending_optimisation = sourceCodeRepository.find_source_codes(source_code_type="SVM", limit=100)

source_code_optimiser = SVMSourceCodeOptimiser()
fail_to_update_source_code = []
# Step2: processing source code from the list one by each time
for source_code_object in source_codes_pending_optimisation:

    source_code_id = source_code_object["_id"]
    source_code_block = source_code_object['source_code'] 

    if source_code_optimiser.optimise(source_code_block):
        source_code_object['status'] = SourceCodeStatus.ABLE_TO_RUN_BO.value
    else:
        source_code_object['status'] = SourceCodeStatus.UNABLE_TO_RUN_BO.value
