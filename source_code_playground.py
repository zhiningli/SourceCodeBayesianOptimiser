from src.preprocessing.CodeStringAnalyser import CodeStrAnalyser



codestr = """
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import openml
import pandas as pd

def run_svm_classification(kernel, C, coef0, gamma):
    # Step 1: Load the dataset
    dataset = openml.datasets.get_dataset(dataset_id=40945)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    y = pd.Series(y, name='target')
    target_names = dataset.retrieve_class_labels() if dataset.retrieve_class_labels() else None

    # Step 2: Drop columns with high cardinality or irrelevant information
    X = X.drop(columns=['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'])

    # Step 3: Encode categorical columns
    # Encode 'sex' column as binary values
    X['sex'] = X['sex'].map({'male': 1, 'female': 0})

    # Encode 'embarked' column with LabelEncoder, filling missing values with the most frequent category
    X['embarked'].fillna(X['embarked'].mode()[0], inplace=True)
    X['embarked'] = LabelEncoder().fit_transform(X['embarked'])

    # Step 4: Impute missing values for the remaining numeric columns
    # Fill missing values in numeric columns with the mean
    numeric_imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(numeric_imputer.fit_transform(X), columns=X.columns)

    # Step 5: Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 6: Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 7: Initialize the SVM model with hyperparameters
    model = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0)

    # Step 8: Train the model
    model.fit(X_train, y_train)

    # Step 9: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 10: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", report)

    return accuracy
"""

code_analyser = CodeStrAnalyser()
code_analyser.extract_information_from_code_string(code_str=codestr)
code_analyser.extract_dataset_from_code_string(code_str=codestr)
code_analyser.perform_statistical_analysis()
print(code_analyser.dataset_statistics)
print(code_analyser.evaluation_metrics)
print(code_analyser.model_statistics)
