import random
import numpy as np
import json
import logging
from enum import Enum

from src.data.data_models import SVMHyperParameterSpace
from src.data.source_code_generator import SVMSourceCode
from src.data.db.source_code_crud import SourceCodeRepository

logging.basicConfig(level=logging.INFO)
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

class SourceCodeStatus(Enum):
    GENERATED_FROM_TEMPLATE = "generated_from_template"
    VALIDATED_TO_RUN = "validated_to_run"


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
                if library != "uci":
                    continue
                # Check if the library is "uci" before building the source code
                if library == "uci":
                    source_code = (
                        SVMSourceCode.builder()
                        .buildDataSet(library=library, dataset_url=dataset)
                        .buildKernel(kernel)
                        .buildC(C)
                        .buildGamma(gamma)
                        .buildCoef0(coef0)
                        .build()
                    )
                    source_codes.append(source_code)
                else:
                    logging.warning(f"Library '{library}' not recognized. Skipping dataset '{dataset}'.")
    return source_codes

# Save batch to the database
def save_batch_to_db(source_codes):
    source_code_repo = SourceCodeRepository()
    try:
        # Convert Enum to string
        status = SourceCodeStatus.GENERATED_FROM_TEMPLATE.value
        source_code_repo.save_source_codes_batch(source_codes, status)
        logging.info("Batch saved successfully")
    except Exception as e:
        logging.error("Failed to save batch: %s", e)


# Main function
def main():
    data_source = load_data_sources()
    source_codes = create_source_codes(data_source, variations=1)
    for code in source_codes:
        print(type(code), code.get_source_code)
    save_batch_to_db(source_codes)

if __name__ == "__main__":
    main()
