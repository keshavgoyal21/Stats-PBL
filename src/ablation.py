from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd


def run_ablation_study(X, X_scaled, y):

    results = []

    # =========================
    # EXP 1: Raw Data (no scaling)
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append([
        "Raw Data",
        "No Scaling",
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average='weighted', zero_division=0)
    ])

    # =========================
    # EXP 2: Scaled Data
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append([
        "Scaled Data",
        "With Scaling",
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average='weighted', zero_division=0)
    ])

    # =========================
    # CREATE TABLE
    # =========================
    df_results = pd.DataFrame(results, columns=[
        "Data", "Scaling", "Accuracy", "F1 Score"
    ])

    print("\n===== ABLATION STUDY =====")
    print(df_results)

    # Save
    df_results.to_csv("outputs/ablation_results.csv", index=False)

    return df_results