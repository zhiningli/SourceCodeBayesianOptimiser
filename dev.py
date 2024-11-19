import logging
from src.data.db.source_code_crud import SourceCodeRepository, SourceCodeStatus
from src.preprocessing.SVM_BO_optimiser import SVM_BO_optimiser
from src.training.bo_metrics.cumulative_regret import cumulative_regret
import numpy as np
from src.data.db.BO_evaluation_crud import BO_evaluation_db
import math

logging.basicConfig(level=logging.INFO)

sourceCodeRepository = SourceCodeRepository()

# Step 1: Load source code with status of generated_from_template
source_code_status = SourceCodeStatus.EVALUATED_BY_SCRIPTS.value

source_code_pending_validation = sourceCodeRepository.find_source_codes(status=source_code_status, limit=15)
optimiser = SVM_BO_optimiser()
db = BO_evaluation_db()
svm_kernel_lengthscale_prior_means = np.arange(0.5, 2.0, 0.5)  # Generates 4
svm_kernel_outputscale_prior_means = np.arange(0.5, 2.0, 0.5)  # Generates 4
svm_C_lengthscale_prior_means = np.arange(0.1, 10.0, 2.5)  # Generates 4
svm_C_outputscale_prior_means = np.arange(0.1, 2.0, 0.5)  # Generates 4
svm_gamma_lengthscale_prior_means = np.arange(0.1, 1.0, 0.5)  # Gener ates 4
svm_gamma_outputscale_prior_means = np.arange(0.1, 2.0, 0.5)  # Generates 4
svm_coef0_lengthscale_prior_means = np.arange(0.1, 0.5, 0.3)  # Generates 4
svm_coef0_outputscale_prior_means = np.arange(0.1, 2.0, 0.5)  # Generates 4

for source_code_object in source_code_pending_validation:
    source_code = source_code_object["source_code"]
    # for kernel_l_mean in svm_kernel_lengthscale_prior_means:
    for kernel_o_mean in svm_kernel_outputscale_prior_means:
            # for C_l_mean in svm_C_lengthscale_prior_means:
        for C_o_mean in svm_C_outputscale_prior_means:
                    # for gamma_l_mean in svm_gamma_lengthscale_prior_means:
            for gamma_o_mean in svm_gamma_outputscale_prior_means:
                            # for coef0_l_mean in svm_coef0_lengthscale_prior_means:
                for coef0_o_mean in svm_coef0_outputscale_prior_means:

                    result_to_save = {
                        "source_code_id": source_code_object["_id"],
                        "BO_hyperparameters": {
                            "svm_kernel_lengthscale_prior_mean": math.sqrt(2),
                            "svm_kernel_outputscale_prior_mean" : kernel_o_mean,
                            "svm_C_lengthscale_prior_mean" : math.sqrt(2),
                            "svm_C_outputscale_prior_mean" : C_o_mean,
                            "svm_gamma_lengthscale_prior_mean": math.sqrt(2),
                            "svm_gamma_outputscale_prior_mean" : gamma_o_mean,
                            "svm_coef0_lengthscale_prior_mean": math.sqrt(2),
                            "svm_coef0_outputscale_prior_mean": coef0_o_mean
                        }
                    }

                    result, best_accuracy, best_SVM_hyperparameters = optimiser.optimise(
                        code_str = source_code,
                        n_iter=50, initial_points=10, sample_per_batch=1,
                        svm_kernel_lengthscale_prior_mean = result_to_save["BO_hyperparameters"]["svm_kernel_lengthscale_prior_mean"],
                        svm_kernel_outputscale_prior_mean = result_to_save["BO_hyperparameters"]["svm_kernel_outputscale_prior_mean"],
                        svm_C_lengthscale_prior_mean = result_to_save["BO_hyperparameters"]["svm_C_lengthscale_prior_mean"],
                        svm_C_outputscale_prior_mean = result_to_save["BO_hyperparameters"]["svm_C_outputscale_prior_mean"],
                        svm_gamma_lengthscale_prior_mean = result_to_save["BO_hyperparameters"]["svm_gamma_lengthscale_prior_mean"],
                        svm_gamma_outputscale_prior_mean = result_to_save["BO_hyperparameters"]["svm_gamma_outputscale_prior_mean"],
                        svm_coef0_lengthscale_prior_mean = result_to_save["BO_hyperparameters"]["svm_coef0_lengthscale_prior_mean"],
                        svm_coef0_outputscale_prior_mean = result_to_save["BO_hyperparameters"]["svm_coef0_outputscale_prior_mean"],
                    )
                    cr = cumulative_regret(result, 1.0) / len(result)
                    result_to_save["normalised_cumulative_regret"] = cr
                    result_to_save["simple_regrets"] = 1.0 - best_accuracy
                    print("best SVM hyperparameters: ", best_SVM_hyperparameters)
                    print("best_accurary: ", best_accuracy)
                    print("normalised_cumulative_regrets: ", cr)


                    db.save_BO_evaluation_results(result_to_save)


    source_code_object["status"] = SourceCodeStatus.BEST_BO_FOUND.value
    sourceCodeRepository.update_source_code(record_id=source_code_object["_id"], update_data=source_code_object)
