from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cybersecurity_traffic.csv"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"

def run_detection():
    df = pd.read_csv(DATA)
    features = ["packet_size","failed_logins","request_frequency","connection_duration",
                "source_bytes","destination_bytes","unique_ports","protocol"]
    rf = joblib.load(MODELS / "random_forest_model.joblib")
    iso = joblib.load(MODELS / "isolation_forest_model.joblib")

    df["classification"] = rf.predict(df[features])
    df["anomaly"] = iso.predict(df[features]) == -1
    df["threat_detected"] = (df["classification"] == 1) | df["anomaly"]
    df["risk_level"] = df["threat_detected"].map({True:"HIGH", False:"LOW"})
    df["classification_name"] = df["classification"].map({0:"Normal", 1:"Threat"})
    df.to_csv(OUT / "predictions.csv", index=False)
    df[df["threat_detected"]].to_csv(OUT / "alerts.csv", index=False)
    return df

if __name__ == "__main__":
    run_detection()
    print("Detection completed.")
