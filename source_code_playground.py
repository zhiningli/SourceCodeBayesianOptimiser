from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn import datasets

def run_svm_classification():
    # Step 1: Load the dataset
    data = datasets.load_iris()  # Corrected dataset loading
    X, y = data.data, data.target
    target_names = data.target_names

    # Step 2: Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 3: Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 4: Initialize the SVM model with hyperparameters
    model = SVC(kernel='poly', C=4.229101316230308, gamma='scale', coef0=0.47320558504355414, random_state=42)

    # Step 5: Train the model
    model.fit(X_train, y_train)

    # Step 6: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 7: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", report)