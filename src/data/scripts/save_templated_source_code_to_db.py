import random
import numpy as np
import json
import logging
from enum import Enum

from src.data.data_models import SVMHyperParameterSpace
from src.data.source_codes.source_code_builder import SVMSourceCode
from src.data.db.source_code_crud import SourceCodeRepository

logging.basicConfig(level=logging.INFO)



class SourceCodeStatus(Enum):
    GENERATED_FROM_TEMPLATE = "generated_from_template"
    VALIDATED_TO_RUN = "validated_to_run"

# Load dataset sources from external JSON configuration
def load_data_sources(filename="src/data/scripts/data_sources.json"):
    with open(filename, "r") as file:
        return json.load(file)

# Generate random hyperparameters
def generate_random_hyperparameters():
    kernel = random.choice(SVMHyperParameterSpace["kernel"]["options"])
    C = np.random.uniform(low=SVMHyperParameterSpace["C"]["range"][0], high=SVMHyperParameterSpace["C"]["range"][1])
    gamma = random.choice(SVMHyperParameterSpace["gamma"]["options"])
    coef0 = np.random.uniform(low=SVMHyperParameterSpace["coef0"]["range"][0], high=SVMHyperParameterSpace["coef0"]["range"][1])
    return kernel, C, gamma, coef0

# Create source code instances for each dataset
def create_source_codes(data_source, variations=4):
    source_codes = []
    for library, datasets in data_source.items():
        for dataset in datasets:
            for _ in range(variations):
                kernel, C, gamma, coef0 = generate_random_hyperparameters()
                
                # Use dataset_name or dataset_id depending on the library
                if library == "sklearn":
                    source_code = (SVMSourceCode.builder()
                                   .buildDataSet(library=library, dataset_name=dataset)
                                   .buildKernel(kernel)
                                   .buildC(C)
                                   .buildGamma(gamma)
                                   .buildCoef0(coef0)
                                   .build())
                elif library == "openml":
                    source_code = (SVMSourceCode.builder()
                                   .buildDataSet(library=library, dataset_id=dataset)
                                   .buildKernel(kernel)
                                   .buildC(C)
                                   .buildGamma(gamma)
                                   .buildCoef0(coef0)
                                   .build())
                
                source_codes.append(source_code)
    return source_codes

# Save batch to the database
def save_batch_to_db(source_codes):
    source_code_repo = SourceCodeRepository()
    try:
        source_code_repo.save_source_codes_batch(source_codes, SourceCodeStatus.GENERATED_FROM_TEMPLATE)
        logging.info("Batch saved successfully")
    except Exception as e:
        logging.error("Failed to save batch: %s", e)

