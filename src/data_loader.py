import pandas as pd

def load_data(path):
    """
    Loads dataset from CSV file.
    """
    return pd.read_csv(path)

def split_features_target(data, target_column):
    """
    Splits dataset into features (X) and target (y).
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y
