import logging

from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.data.source_code_validator import SourceCodeValidator

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

# Step 1: Load source code with status of generated_from_template
source_code_status = SourceCodeStatus.GENERATED_FROM_TEMPLATE.value

source_codes_pending_validation = sourceCodeRepository.find_source_codes(status=source_code_status, limit = 20)

source_code_validator = SourceCodeValidator()
fail_to_update_source_code = []
# Step2: processing source code from the list one by each time
for source_code_object in source_codes_pending_validation:

    source_code_id = source_code_object["_id"]
    source_code_block = source_code_object['source_code'] 

    validated_source_code = source_code_validator.iterative_refinement_and_validation(source_code_block)
    if validated_source_code:
        source_code_object['source_code'] = validated_source_code
        source_code_object['status'] = SourceCodeStatus.VALIDATED_TO_RUN.value
        sourceCodeRepository.update_source_code(record_id=source_code_id, update_data= source_code_object)
    else:
        logging.warning(f"source code validation for source code {source_code_id} has failed")
        fail_to_update_source_code.append(source_code_object)
        source_code_object['source_code'] = validated_source_code
        source_code_object['status'] = SourceCodeStatus.FAILED_VALIDATION.value
        sourceCodeRepository.update_source_code(record_id=source_code_id, update_data= source_code_object)


print(fail_to_update_source_code)