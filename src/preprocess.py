import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path):
    """
    Load dataset from given path
    """
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """
    Clean dataset:
    - Fix column names
    - Handle infinite values
    - Handle missing values
    - Remove duplicates
    - Drop irrelevant columns
    """

    # 🔹 Remove extra spaces in column names
    df.columns = df.columns.str.strip()

    # 🔹 Replace infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 🔹 Fill NaN values with column mean (better than dropping)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 🔹 Remove duplicates
    df.drop_duplicates(inplace=True)

    # 🔹 Drop irrelevant columns (if present)
    df.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'],
            axis=1, inplace=True, errors='ignore')

    return df


def encode_data(df):
    """
    Encode categorical columns (if any)
    """
    le = LabelEncoder()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df


def split_features_target(df):
    """
    Split dataset into features (X) and target (y)
    """

    # Ensure column names are clean
    df.columns = df.columns.str.strip()

    if 'Label' not in df.columns:
        raise Exception("Target column 'Label' not found. Check dataset.")

    X = df.drop('Label', axis=1)
    y = df['Label']

    return X, y


def scale_features(X):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled