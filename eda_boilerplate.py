import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV data into a pandas DataFrame.
    
    Steps:
    - Read the CSV from the given filepath.
    - Return the DataFrame.
    """
    # TODO: Implement loading the CSV (e.g., pd.read_csv(filepath))
    pass


def summarize_data(df: pd.DataFrame):
    """
    Print basic summary statistics and info about the dataset.
    
    Steps:
    - Print df.info() to see column types and missing values.
    - Print df.describe() to get summary stats on numeric features.
    - Check for missing values (df.isnull().sum()) and print the results.
    """
    # TODO: Implement summary statistics and info printing
    pass


def plot_distributions(df: pd.DataFrame):
    """
    Plot distributions for numeric and categorical features.
    
    Steps:
    - Identify numeric columns and plot histograms/KDE plots.
    - Identify categorical columns and plot bar charts of value counts.
    - Consider using seaborn (sns.histplot, sns.countplot) or matplotlib for simplicity.
    - Show or save the plots for EDA purposes.
    """
    # TODO: Implement plotting distributions for numeric and categorical features
    pass


def analyze_relationships(df: pd.DataFrame):
    """
    Analyze relationships between features, and between features and the target.
    
    Steps:
    - If there's a target variable (e.g., loan_status), consider encoding it numerically
      (e.g., approved=1, denied=0) for correlation analysis.
    - Create a correlation heatmap for numeric features (e.g., using sns.heatmap).
    - Plot boxplots or violin plots of numeric features grouped by the target to see patterns.
    """
    # TODO: Implement relationship analysis (correlations, boxplots by target)
    pass


if __name__ == "__main__":
    df = load_data("mortgage_training_data.csv")
    summarize_data(df)
    plot_distributions(df)
    analyze_relationships(df)

    pass
