from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


os.makedirs("outputs/plots", exist_ok=True)


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)


    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using success metrics
    """

    y_pred = model.predict(X_test)

  
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n MODEL PERFORMANCE ")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

  
    print("\n CLASSIFICATION REPORT")
    report = classification_report(y_test, y_pred)
    print(report)

  
    with open("outputs/metrics.txt", "w") as f:
        f.write("MODEL PERFORMANCE \n")
        f.write(f"Accuracy  : {acc}\n")
        f.write(f"Precision : {precision}\n")
        f.write(f"Recall    : {recall}\n")
        f.write(f"F1 Score  : {f1}\n\n")
        f.write(" CLASSIFICATION REPORT \n")
        f.write(report)

   
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.show()

    return acc, precision, recall, f1