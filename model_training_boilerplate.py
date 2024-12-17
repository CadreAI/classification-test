import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV data into a pandas DataFrame.
    
    Steps:
    - Use pd.read_csv(filepath) to load the data.
    - Return the DataFrame.
    """
    # TODO: Implement loading of CSV file
    pass


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the DataFrame:
    - Separate the target (loan_status) from the features.
    - Convert loan_status to numeric: approved=1, denied=0.
    - Drop non-predictive columns (e.g., application_id).
    - Identify categorical columns and one-hot encode them.
    - Return feature matrix (X) and target vector (y).
    """
    # TODO: Implement preprocessing steps described above
    pass


def train_and_evaluate(X, y):
    """
    Train a model and evaluate its performance:
    - Split data into train/test sets.
    - Initialize an XGBoost model.
    - Fit the model on the training set.
    - Predict on the test set and compute accuracy and AUC.
    - Print out the evaluation metrics.
    """
    # TODO: Implement train/test split
    # TODO: Initialize and train the XGBClassifier
    # TODO: Predict and compute accuracy and AUC
    # TODO: Print evaluation metrics
    pass


if __name__ == "__main__":
    # Load data
    df = load_data("mortgage_training_data.csv")

    # Preprocess data
    X, y = preprocess_data(df)

    # Train and evaluate the model
    train_and_evaluate(X, y)
