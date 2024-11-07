import logging
from datetime import datetime
from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.data.source_code_optimiser import SVMSourceCodeOptimiser

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

# Step 1: Load source code with status of generated_from_template
source_code_status = SourceCodeStatus.VALIDATED_TO_RUN.value

source_codes_pending_optimisation = sourceCodeRepository.find_source_codes(status=source_code_status, limit=100)

source_code_optimiser = SVMSourceCodeOptimiser()
fail_to_update_source_code = []
# Step2: processing source code from the list one by each time
for source_code_object in source_codes_pending_optimisation:

    source_code_id = source_code_object["_id"]
    source_code_block = source_code_object['source_code'] 

    best_accuracy, best_kernel, best_C, best_coef0, best_gamma = source_code_optimiser.optimise(source_code_block, n_iter=100)
    source_code_object['status'] = SourceCodeStatus.ABLE_TO_RUN_BO.value

    source_code_object['source_code_hyperparameters']['kernel'] = best_kernel
    source_code_object['source_code_hyperparameters']['C'] = best_C
    source_code_object['source_code_hyperparameters']['coef0'] = best_coef0
    source_code_object['source_code_hyperparameters']['gamma'] = best_gamma
    source_code_object['source_code_hyperparameters']['accurary'] = best_accuracy

    sourceCodeRepository.update_source_code(record_id=source_code_id, update_data=source_code_object) 