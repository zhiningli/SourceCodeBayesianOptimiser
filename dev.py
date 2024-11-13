import logging
from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.preprocessing.SVM_BO_optimiser import SVM_BO_optimiser
from src.training.bo_metrics.cumulative_regret import cumulative_regret
import numpy as np

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

# Step 1: Load source code with status of generated_from_template
source_code_status = SourceCodeStatus.FAILED_VALIDATION.value

source_code_pending_validation = sourceCodeRepository.get_source_code(name="SVM_source_code_on_openml_40945_dataset")

source_code = source_code_pending_validation["source_code"]
optimiser = SVM_BO_optimiser()


result, best_accuracy, best_SVM_hyperparameters = optimiser.optimise(
    code_str = source_code,
    n_iter=20, initial_points=10, sample_per_batch=1,
    svm_kernel_lengthscale_prior_mean = 1,
    svm_kernel_outputscale_prior_mean = 1,
    svm_C_lengthscale_prior_mean = 1.5,
    svm_C_outputscale_prior_mean = 1.5,
    svm_gamma_lengthscale_prior_mean = 1,
    svm_gamma_outputscale_prior_mean = 1,
    svm_coef0_lengthscale_prior_mean = 1.5,
    svm_coef0_outputscale_prior_mean = 1
)

print(result)
iteration_values_converted = np.array([t.item() for t in result], dtype=np.float64)

cr = cumulative_regret(iteration_values_converted, 1.0) / len(iteration_values_converted)

print("best SVM hyperparameters: ", best_SVM_hyperparameters)
print("best_accurary: ", best_accuracy)
print("normalised_cumulative_regrets: ", cr)
