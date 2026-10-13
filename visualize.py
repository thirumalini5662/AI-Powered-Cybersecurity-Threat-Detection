from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cybersecurity_traffic.csv"
OUT = ROOT / "outputs"
IMG = ROOT / "images"

def create_visualizations():
    IMG.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    pred = pd.read_csv(OUT / "test_predictions.csv")

    plt.figure(figsize=(7,5))
    df["label_name"].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class"); plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(IMG / "class_distribution.png", dpi=150); plt.close()

    plt.figure(figsize=(6,5))
    ConfusionMatrixDisplay.from_predictions(pred["actual"], pred["predicted"])
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(IMG / "confusion_matrix.png", dpi=150); plt.close()

    # Simple importance visualization using correlations for an easy-to-explain chart.
    nums = ["packet_size","failed_logins","request_frequency","connection_duration",
            "source_bytes","destination_bytes","unique_ports"]
    corr = df[nums + ["attack_label"]].corr(numeric_only=True)["attack_label"].drop("attack_label").abs().sort_values()
    plt.figure(figsize=(7,5)); corr.plot(kind="barh")
    plt.title("Feature Relationship with Threat Label")
    plt.xlabel("Absolute correlation"); plt.tight_layout()
    plt.savefig(IMG / "feature_importance.png", dpi=150); plt.close()

    plt.figure(figsize=(7,5))
    normal = df[df.attack_label == 0]
    threat = df[df.attack_label == 1]
    plt.scatter(normal["request_frequency"], normal["packet_size"], s=8, label="Normal")
    plt.scatter(threat["request_frequency"], threat["packet_size"], s=8, label="Threat")
    plt.title("Traffic Anomaly / Threat Pattern")
    plt.xlabel("Request Frequency"); plt.ylabel("Packet Size"); plt.legend()
    plt.tight_layout(); plt.savefig(IMG / "anomaly_scatter.png", dpi=150); plt.close()

if __name__ == "__main__":
    create_visualizations()
