from src.preprocess import *
from src.eda import *
from src.model import train_model, evaluate_model
from src.advancemodel import run_random_forest   
from src.ablation import run_ablation_study
from src.erroranalysis import error_analysis      


def main():

    
    df = load_data("data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

    
    df = clean_data(df)
    df = encode_data(df)

    X, y = split_features_target(df)
    X_scaled = scale_features(X)


    model, X_test, y_test = train_model(X_scaled, y)
    lr_acc, lr_precision, lr_recall, lr_f1 = evaluate_model(model, X_test, y_test)

    
    error_analysis(model, X_test, y_test)
            
    rf_acc, rf_precision, rf_recall, rf_f1 = run_random_forest(X_scaled, y)

       
    print("\n MODEL COMPARISON ")

    print("\nLogistic Regression:")
    print(f"Accuracy  : {lr_acc:.4f}, F1: {lr_f1:.4f}")

    print("\nRandom Forest:")
    print(f"Accuracy  : {rf_acc:.4f}, F1: {rf_f1:.4f}")


   
    run_ablation_study(X, X_scaled, y)


if __name__ == "__main__":
    main()