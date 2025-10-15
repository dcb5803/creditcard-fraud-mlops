# src/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from utils import preprocess

mlflow.set_tracking_uri("file:///mlruns")
mlflow.set_experiment("CreditCardFraud")

def main():
    df = pd.read_csv("data/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess(df)

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")

        print(f"ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
