import random
import numpy as np
import json
import logging
from enum import Enum

from src.data.data_models import SVMHyperParameterSpace
from src.data.source_code_generator import SVMSourceCode
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

    source_code = (SVMSourceCode.builder()
                    .buildDataSet(library="UCI", dataset_url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
                    .buildKernel("rbf")
                    .buildC(0.5)
                    .buildGamma("auto")
                    .buildCoef0(0.5)
                    .build())

            
    return source_code



# Main function
def main():
    data_source = load_data_sources()
    source_code = create_source_codes(data_source, variations=4)   

    print(source_code.get_source_code)

if __name__ == "__main__":
    main()
