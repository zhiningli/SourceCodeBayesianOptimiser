import logging
from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.data.source_code_validator import SourceCodeValidator

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

# Step 1: Load source code with status of generated_from_template
source_code_status = SourceCodeStatus.BEST_BO_FOUND.value

source_codes_pending_validation = sourceCodeRepository.find_source_codes(status=source_code_status, limit = 300)

# Step2: processing source code from the list one by each time
for source_code_object in source_codes_pending_validation:

    source_code_id = source_code_object["_id"]
    source_code_object["status"] = SourceCodeStatus.EVALUATED_BY_SCRIPTS.value
    sourceCodeRepository.update_source_code(record_id=source_code_id, update_data=source_code_object)