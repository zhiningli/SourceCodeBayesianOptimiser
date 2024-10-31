from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn import datasets
import pandas as pd

def run_svm_classification(container_path=None):
    # Step 1: Load the dataset
    data = datasets.load_iris()  # Assume Iris dataset, replace with your actual dataset
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    target_names = [str(name) for name in data.target_names]

    # Step 2: Preprocess the dataset
    # Encode categorical columns
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Fill missing values
    X.fillna(X.mean(), inplace=True)

    # Step 3: Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 4: Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 5: Initialize the SVM model with hyperparameters
    model = SVC(kernel='sigmoid', C=1.7606049924528766, gamma='scale', coef0=0.7510997120472259, random_state=42)

    # Step 6: Train the model
    model.fit(X_train, y_train)

    # Step 7: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 8: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", report)

run_svm_classification()