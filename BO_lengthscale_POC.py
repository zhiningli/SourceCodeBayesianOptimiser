from src.preprocessing.SVM_BO_optimiser import SVM_BO_optimiser
from math import sqrt
import numpy as np

source_code_str = """
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import openml
import pandas as pd


def run_svm_classification(kernel, C, coef0, gamma):
    # Step 1: Load the dataset
    
    dataset = openml.datasets.get_dataset(dataset_id=42803)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    y = pd.Series(y, name='target')
    target_names = dataset.retrieve_class_labels() if dataset.retrieve_class_labels() else None


    # Limit dataset to top 10,000 samples if larger
    max_samples = 10000
    if len(X) > max_samples:
        X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y)

    # Step 2: Preprocess the dataset
    # Fill missing values only for numerical columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

    # Encode categorical columns
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])
        
    # Step 3: Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 4: Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 5: Initialize the SVM model with hyperparameters
    model = SVC(kernel=kernel, C = C, gamma= gamma, coef0 = coef0)

    # Step 6: Train the model
    model.fit(X_train, y_train)

    # Step 7: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 8: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", accuracy)

    return accuracy
"""

optimiser = SVM_BO_optimiser()

lengthscale_prior_means = np.linspace(0.1, 4, 40)

results_for_plotting = {}
for lengthscale_prior_mean in lengthscale_prior_means:
    
    averaged_y = 0
    best_y = 0
    for _ in range(5):
        result, best_accuracy, best_SVM_hyperparameters = optimiser.optimise(
            code_str = source_code_str,
            n_iter=50, initial_points=10, sample_per_batch=1,
            svm_kernel_lengthscale_prior_mean = lengthscale_prior_mean,
            svm_kernel_outputscale_prior_mean = sqrt(2),
            svm_C_lengthscale_prior_mean = sqrt(2),
            svm_C_outputscale_prior_mean = sqrt(2),
            svm_gamma_lengthscale_prior_mean = sqrt(2),
            svm_gamma_outputscale_prior_mean = sqrt(2),
            svm_coef0_lengthscale_prior_mean = sqrt(2),
            svm_coef0_outputscale_prior_mean = sqrt(2),
        )

        if best_accuracy >= best_y:
            best_y = best_accuracy
        averaged_y += best_y
    results_for_plotting[lengthscale_prior_mean] = averaged_y / 5

print(results_for_plotting)