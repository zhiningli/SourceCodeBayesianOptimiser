import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn import datasets

def extract_datasets():
    # Load the dataset
    data = datasets.load_iris()
    X, y = data.data, data.target
    target_names = data.target_names

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert the data to pandas DataFrames
    X_train = pd.DataFrame(X_train, columns=data.feature_names)
    X_test = pd.DataFrame(X_test, columns=data.feature_names)

    return X_train, y_train, X_test, y_test

# Execute the function
extract_datasets()
