from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os


os.makedirs("outputs/plots", exist_ok=True)


def run_random_forest(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)


    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n RANDOM FOREST PERFORMANCE")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\n CLASSIFICATION REPORT ")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)


    with open("outputs/rf_metrics.txt", "w") as f:
        f.write(" RANDOM FOREST PERFORMANCE \n")
        f.write(f"Accuracy  : {acc}\n")
        f.write(f"Precision : {precision}\n")
        f.write(f"Recall    : {recall}\n")
        f.write(f"F1 Score  : {f1}\n\n")
        f.write(report)


    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("outputs/plots/rf_confusion_matrix.png")
    plt.show()

    return acc, precision, recall, f1
