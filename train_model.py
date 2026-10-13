from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cybersecurity_traffic.csv"
MODEL_DIR = ROOT / "models"
OUT = ROOT / "outputs"

def train_models():
    df = pd.read_csv(DATA)
    features = ["packet_size","failed_logins","request_frequency","connection_duration",
                "source_bytes","destination_bytes","unique_ports","protocol"]
    X, y = df[features], df["attack_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("protocol", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["protocol"])
    ], remainder="passthrough")

    rf = Pipeline([("preprocessor", pre), ("model", RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced"))])
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred, zero_division=0)
    }

    iso = Pipeline([("preprocessor", pre), ("model", IsolationForest(
        n_estimators=150, contamination=.28, random_state=42))])
    iso.fit(X_train)

    MODEL_DIR.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    joblib.dump(rf, MODEL_DIR / "random_forest_model.joblib")
    joblib.dump(iso, MODEL_DIR / "isolation_forest_model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "classification_report.txt").write_text(classification_report(y_test, pred))
    pd.DataFrame({"actual": y_test, "predicted": pred}).to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, pred)).to_csv(OUT / "confusion_matrix.csv", index=False)
    return metrics

if __name__ == "__main__":
    print(train_models())
