import logging
from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.data.source_code_validator import SourceCodeValidator

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

# Step 1: Load source code with status of generated_from_template
source_code_status = SourceCodeStatus.FAILED_VALIDATION.value

source_codes_pending_validation = sourceCodeRepository.find_source_codes(status=source_code_status, limit = 300)

source_code_validator = SourceCodeValidator()
fail_to_update_source_code = []
# Step2: processing source code from the list one by each time
for source_code_object in source_codes_pending_validation:

    source_code_id = source_code_object["_id"]
    sourceCodeRepository.delete_source_code(source_code_id)