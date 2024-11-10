from src.preprocessing.CodeStringAnalyser import CodeStrAnalyser
import time
import logging
from datetime import datetime
from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

source_code_status = SourceCodeStatus.ABLE_TO_RUN_BO.value

source_codes_pending_optimisation = sourceCodeRepository.find_source_codes(status=source_code_status, limit=1)


fail_to_evaluate_source_code = []
# Step2: processing source code from the list one by each time
for source_code_object in source_codes_pending_optimisation:
    code_analyser = CodeStrAnalyser()
    source_code_id = source_code_object["_id"]
    source_code_block = source_code_object['source_code'] 
    if code_analyser.extract_information_from_code_string(code_str=source_code_block):
        time.sleep(5.0)
    else:
        fail_to_evaluate_source_code.append(source_code_object)

    if code_analyser.extract_dataset_from_code_string(code_str=source_code_block):
        code_analyser.perform_statistical_analysis()

  
        source_code_object["dataset_statistics"] = code_analyser.dataset_statistics
        source_code_object["model_statistics"] = code_analyser.model_statistics
        source_code_object["evaluation_metrics"] = code_analyser.evaluation_metrics
        logging.info(source_code_object)
        source_code_object['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_code_object['status'] = SourceCodeStatus.EVALUATED_BY_SCRIPTS.value

        sourceCodeRepository.update_source_code(source_code_id, source_code_object)
    else:
        fail_to_evaluate_source_code.append(source_code_object)
    
