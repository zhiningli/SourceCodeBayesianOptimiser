import numpy as np
import logging
from enum import Enum

from src.data.data_models import SVMHyperParameterSpace
from src.data.source_code_generator import SVMSourceCode
from src.data.db.source_code_crud import SourceCodeRepository

logging.basicConfig(level=logging.INFO)

class SourceCodeStatus(Enum):
    GENERATED_FROM_TEMPLATE = "generated_from_template"
    VALIDATED_TO_RUN = "validated_to_run"


