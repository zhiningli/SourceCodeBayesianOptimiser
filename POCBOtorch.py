import torch
from sklearn.model_selection import cross_val_score
from botorch.models import MixedSingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.fit import fit_gpytorch_mll_torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn import datasets
import numpy as np

from sklearn import datasets
data = datasets.load_iris()
X, y = data.data, data.target

kernel_options = ["linear", "poly", "rbf", "sigmoid"]
gamma_options = ["scale", "auto"]

def run_svm_classification(kernel = "rbf", C = 1.0, gamma = "auto", coef0 = 0.4):

    data = datasets.load_iris()
    X, y = data.data, data.target
    target_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel=kernel, C = C, gamma= gamma, coef0 = coef0, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", report)

    return accuracy

# Update botorch_objective to interpret kernel and gamma as categorical variables
def botorch_objective(params):
    np_params = params.detach().numpy().flatten()  # Flatten to ensure 1D array for easy indexing
    kernel_idx = int(np.round(np_params[0]))       # Use np.round to get nearest integer for kernel
    gamma_idx = int(np.round(np_params[3]))        # Same for gamma

    # Map indices to categorical values
    kernel = kernel_options[kernel_idx]
    gamma = gamma_options[gamma_idx]
    C = np_params[1]
    coef0 = np_params[2]

    # Run the SVM classification with the selected hyperparameters
    result = run_svm_classification(kernel=kernel, C=C, gamma=gamma, coef0=coef0)
    return torch.tensor(result)

# Define bounds for the input parameters
bounds = torch.tensor([[0, 0.1, 0.0, 0], [len(kernel_options) - 1, 10.0, 1.0, len(gamma_options) - 1]])

# Step 1: Initial Sample Points for Training
train_x = draw_sobol_samples(bounds=bounds, n=10, q=1).squeeze(1)
train_x = (train_x - bounds[0]) / (bounds[1] - bounds[0])
unnormalized_train_x = train_x * (bounds[1] - bounds[0]) + bounds[0]

# Ensure initial train_y is of shape [10, 1]
train_y = torch.tensor([botorch_objective(x) for x in unnormalized_train_x]).unsqueeze(-1)
train_y = standardize(train_y).view(-1, 1)  # Explicitly reshape to [n, 1] if needed

# Confirm train_y shape after standardization
print("Initial train_y shape:", train_y.shape)  # Expected to be [10, 1]

# Initialize the GP model
gp = MixedSingleTaskGP(train_x, train_y, cat_dims=[0, 3])
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

# Fit the GP model
fit_gpytorch_mll_torch(mll)

# Define the Acquisition Function
ei = LogExpectedImprovement(model=gp, best_f=train_y.max())


# Optimization Loop
for i in range(20):  # Number of BO iterations
    candidate, _ = optimize_acqf(
        acq_function=ei,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )


    train_y = train_y.view(-1, 1)   
    # Evaluate the candidate and ensure new_y has shape [1, 1]
    new_y = botorch_objective(candidate).view(1, 1)  # Ensure new_y shape [1, 1]
    # Concatenate the new candidate and new_y to training data
    train_x = torch.cat([train_x, candidate.view(1, -1)])  # Ensure train_x matches shape
    train_y = torch.cat([train_y, new_y], dim=0)  # Concatenate along dimension 0
    train_y = train_y.view(-1)


    # Update GP model with new training data
    gp.set_train_data(inputs=train_x, targets=train_y, strict=False)  # Use strict=False if needed


    # Update acquisition function
    ei = LogExpectedImprovement(model=gp, best_f=train_y.max())

# Best Hyperparameters
best_idx = train_y.argmax()
best_hyperparams = train_x[best_idx]

# Decoding best parameters
best_kernel = kernel_options[int(best_hyperparams[0].item())]
best_C = best_hyperparams[1].item()
best_coef0 = best_hyperparams[2].item()
best_gamma = gamma_options[int(best_hyperparams[3].item())]

print("Best Hyperparameters:")
print("Kernel:", best_kernel)
print("C:", best_C)
print("Coef0:", best_coef0)
print("Gamma:", best_gamma)
print("Accuracy:", run_svm_classification(best_kernel, best_C, best_gamma, best_coef0))
