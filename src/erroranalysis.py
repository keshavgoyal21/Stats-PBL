import numpy as np


def error_analysis(model, X_test, y_test):

    # Predictions
    y_pred = model.predict(X_test)

    # ================================
    # ❌ MISCLASSIFIED SAMPLES
    # ================================
    misclassified = np.where(y_test != y_pred)[0]

    total = len(y_test)
    wrong = len(misclassified)

    print("\n===== ERROR ANALYSIS =====")
    print(f"Total Samples       : {total}")
    print(f"Misclassified       : {wrong}")
    print(f"Error Rate          : {wrong/total:.4f}")
    print(f"Accuracy (check)    : {(1 - wrong/total):.4f}")

    # ================================
    # 🔍 SHOW SAMPLE ERRORS
    # ================================
    print("\nSample Misclassifications:")
    for i in misclassified[:10]:   # show first 10
        print(f"Actual: {y_test.iloc[i]} | Predicted: {y_pred[i]}")