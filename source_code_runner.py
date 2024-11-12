from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
        
def load_dataset_from_url(data_url, column_names=None, target_column=None):
    # Load the dataset
    data = pd.read_csv(data_url, header=None if column_names else 'infer')
    if column_names:
        data.columns = column_names
    
    # Separate features and target
    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X = data.iloc[:, :-1]  # Use all columns except the last as features
        y = data.iloc[:, -1]   # Use the last column as the target

    return X, y


def run_svm_classification(kernel, C, coef0, gamma):
    # Step 1: Load the dataset
    
    X, y = load_dataset_from_url(data_url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    target_names = label_encoder.classes_


    # Limit dataset to top 10,000 samples if larger
    max_samples = 10000
    if len(X) > max_samples:
        X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=42)

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
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", report)

    return accuracy
run_svm_classification(kernel="sigmoid", C=0.5, gamma="auto", coef0=1)
