from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
OUTPUT_DIR = BASE / "outputs"
FEATURES = ["packet_size","failed_logins","request_frequency","connection_duration",
            "source_bytes","destination_bytes","unique_ports","protocol"]

st.set_page_config(page_title="AI Cybersecurity Threat Detection", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_models():
    return (joblib.load(MODEL_DIR/"random_forest_model.joblib"),
            joblib.load(MODEL_DIR/"isolation_forest_model.joblib"))

def predict(values):
    rf, iso = load_models()
    row = pd.DataFrame([values], columns=FEATURES)
    classification = "Threat" if int(rf.predict(row)[0]) == 1 else "Normal"
    anomaly = int(iso.predict(row)[0]) == -1
    threat = classification == "Threat" or anomaly
    return classification, anomaly, threat, ("HIGH" if threat else "LOW")

st.title("🛡️ AI-Powered Cybersecurity Threat Detection System")
st.caption("Machine-learning project using simulated network traffic.")

page = st.sidebar.radio("Choose a page", ["Dashboard", "Live Prediction", "Project Information"])

if page == "Dashboard":
    st.header("Project Dashboard")
    mf = OUTPUT_DIR/"metrics.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        a,b,c,d = st.columns(4)
        a.metric("Accuracy", f"{m['accuracy']:.2%}")
        b.metric("Precision", f"{m['precision']:.2%}")
        c.metric("Recall", f"{m['recall']:.2%}")
        d.metric("F1 Score", f"{m['f1_score']:.2%}")
    for title, fn in [
        ("Class Distribution","class_distribution.png"),
        ("Confusion Matrix","confusion_matrix.png"),
        ("Feature Importance","feature_importance.png"),
        ("Anomaly Detection","anomaly_scatter.png")]:
        p = BASE/"images"/fn
        if p.exists():
            st.subheader(title); st.image(str(p), use_container_width=True)
    af = OUTPUT_DIR/"alerts.csv"
    if af.exists():
        st.subheader("Threat Alerts")
        st.dataframe(pd.read_csv(af).head(20), use_container_width=True)

elif page == "Live Prediction":
    st.header("🔎 Live Network Traffic Prediction")
    c1,c2 = st.columns(2)
    with c1:
        packet_size = st.number_input("Packet Size", min_value=1.0, value=700.0)
        failed_logins = st.number_input("Failed Logins", min_value=0.0, value=0.0)
        request_frequency = st.number_input("Request Frequency", min_value=0.0, value=30.0)
        connection_duration = st.number_input("Connection Duration", min_value=.1, value=12.0)
    with c2:
        source_bytes = st.number_input("Source Bytes", min_value=0.0, value=5000.0)
        destination_bytes = st.number_input("Destination Bytes", min_value=0.0, value=8000.0)
        unique_ports = st.number_input("Unique Ports", min_value=1.0, value=2.0)
        protocol = st.selectbox("Protocol", ["HTTP","HTTPS","TCP","UDP","ICMP"])
    if st.button("🚨 Analyze Traffic", type="primary"):
        values = {"packet_size":packet_size,"failed_logins":failed_logins,
                  "request_frequency":request_frequency,"connection_duration":connection_duration,
                  "source_bytes":source_bytes,"destination_bytes":destination_bytes,
                  "unique_ports":unique_ports,"protocol":protocol}
        cls, anomaly, threat, risk = predict(values)
        a,b,c,d = st.columns(4)
        a.metric("Classification", cls)
        b.metric("Anomaly", "Yes" if anomaly else "No")
        c.metric("Threat Detected", "Yes" if threat else "No")
        d.metric("Risk Level", risk)
        if threat: st.error("⚠️ Potential cybersecurity threat detected.")
        else: st.success("✅ Traffic appears normal.")

else:
    st.header("📘 Project Information")
    st.markdown("""
**Objective:** Detect suspicious network traffic using machine learning.

**Models:** Random Forest Classifier and Isolation Forest.

**Simulated threats:** Brute Force, DoS, Port Scan and Data Exfiltration.

**Technology:** Python, Pandas, NumPy, Scikit-learn, Matplotlib and Streamlit.

This is an educational project using simulated data. It is not a production IDS/SOC.
""")
