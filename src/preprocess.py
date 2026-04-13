import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path):
    
    df = pd.read_csv(path)
    return df


def clean_data(df):
    
    df.columns = df.columns.str.strip()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.fillna(df.mean(numeric_only=True), inplace=True)

    df.drop_duplicates(inplace=True)

    df.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'],
            axis=1, inplace=True, errors='ignore')

    return df


def encode_data(df):
    
    le = LabelEncoder()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df


def split_features_target(df):
    

    
    df.columns = df.columns.str.strip()

    if 'Label' not in df.columns:
        raise Exception("Target column 'Label' not found. Check dataset.")

    X = df.drop('Label', axis=1)
    y = df['Label']

    return X, y


def scale_features(X):
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled