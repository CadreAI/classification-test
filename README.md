### Predictive Modeling on Mortgage Loan Tabular Data

**Overview:**  
You have been given a tabular dataset representing processed mortgage application data. Each row corresponds to a mortgage application, along with various features such as annual income, loan amount, and a binary outcome indicating whether the loan was approved. Your task is to build a simple predictive model that predicts whether a loan will be approved or denied, given the features.

**Data Description (mortgage_training_data.csv):**  
Assume you have a CSV file with the following columns (already cleaned):

- `application_id` (int): unique identifier (not a predictor, but a key)
- `annual_income` (float)
- `loan_amount` (float)
- `debt_to_income_ratio` (float)
- `region` (string): categorical feature, one of {“West”, “East”, “North”, “South”}
- `loan_status` (string): target variable, either “approved” or “denied”

**What You Need to Do:**

1. **Load the Data:**
   - Load `mortgage_training_data.csv` into a Pandas DataFrame.
   
2. **Exploratory Data Analysis (EDA) (Option in person / Required if takehome):**
   - Quickly check for missing values or anomalies.
   - Print basic summary statistics.
   
   *(You are not required to do extensive EDA due to time, but a brief check is good.)*

3. **Feature Engineering:**
   - Convert categorical variables into numeric form. For `region`, you can use one-hot encoding or an equivalent method.
   - Ensure all features are numeric and suitable for modeling.
   
4. **Split the Data:**
   - Separate `loan_status` as the target (`y`) and the remaining relevant columns as features (`X`).
   - Perform a train/test split (e.g., 80% train, 20% test) using `sklearn.model_selection.train_test_split`.

5. **Model Training:**
   - Train a simple classification model to predict `loan_status`.
   - You may choose any reasonable model (e.g., Logistic Regression, Random Forest, or XGBoost).
   - Fit the model on the training set.

6. **Model Evaluation:**
   - Predict on the test set and compute evaluation metrics:
     - Accuracy score
     - AUC (Area Under the ROC Curve) if applicable
   - Print these metrics.

7. **Model Interpretation (Optional):**
   - (If time allows) Print feature importances for tree-based models or coefficients for linear models to understand what features are most predictive.

8. **Deliverables:**
   - A Python script (`model_training.py`) that:
     - Loads the data
     - Prepares the features/target
     - Trains a model
     - Evaluates and prints metrics
   
   - There’s no need to save the model to disk for this exercise unless you wish to do so.

**What We’re Looking For:**
- Proper handling of categorical features.
- Correct train/test split to avoid data leakage.
- Correct fitting and prediction with a standard ML model.
- Computation and interpretation of basic metrics (accuracy, AUC).

**Time Limit:**
We expect you to complete this within an hour. If you have any questions, feel free to ask.

---

### Boilerplate Code (Optional Starting Point)

```python
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
```
