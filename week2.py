from src.preprocess import *
from src.eda import *
from src.model import train_model, evaluate_model


def main():
    # ================================
    # 📂 LOAD DATA
    # ================================
    df = load_data("data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

    # ================================
    # 🧹 PREPROCESSING
    # ================================
    df = clean_data(df)
    df = encode_data(df)

    X, y = split_features_target(df)
    X_scaled = scale_features(X)

    # ================================
    # 📊 EDA
    # ================================
    descriptive_stats(df)
    class_distribution(y)
    correlation_heatmap(df)
    feature_distribution(df)
    boxplot_outliers(df)

    # ================================
    # 🤖 MODEL (LOGISTIC REGRESSION)
    # ================================
    model, X_test, y_test = train_model(X_scaled, y)

    acc, precision, recall, f1 = evaluate_model(model, X_test, y_test)

    # ================================
    # 📈 FINAL METRICS PRINT
    # ================================
    print("\n===== FINAL SUMMARY =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


# ================================
# 🚀 RUN SCRIPT
# ================================
if __name__ == "__main__":
    main()