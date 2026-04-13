from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


os.makedirs("outputs/plots", exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name="model"):

    y_pred = model.predict(X_test)

  
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n {model_name.upper()} PERFORMANCE")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

   
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\nCLASSIFICATION REPORT")
    print(report)

    
    with open(f"outputs/{model_name}_metrics.txt", "w") as f:
        f.write(f"{model_name.upper()} PERFORMANCE\n")
        f.write(f"Accuracy  : {acc}\n")
        f.write(f"Precision : {precision}\n")
        f.write(f"Recall    : {recall}\n")
        f.write(f"F1 Score  : {f1}\n\n")
        f.write(report)

   
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"outputs/plots/{model_name}_cm.png")
    plt.show()

    return acc, precision, recall, f1
