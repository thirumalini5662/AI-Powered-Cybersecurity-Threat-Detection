from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cybersecurity_traffic.csv"

def generate_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    protocols = ["HTTP", "HTTPS", "TCP", "UDP", "ICMP"]

    for i in range(n):
        attack = rng.random() < 0.28
        if not attack:
            packet = rng.normal(700, 180)
            failed = rng.poisson(0.4)
            freq = rng.normal(35, 12)
            duration = rng.normal(12, 4)
            src = rng.normal(6000, 1800)
            dst = rng.normal(8000, 2200)
            ports = max(1, int(rng.normal(3, 2)))
            label = 0
            name = "Normal"
        else:
            kind = rng.choice(["BruteForce", "DoS", "PortScan", "DataExfiltration"])
            label = 1
            name = kind
            if kind == "BruteForce":
                packet = rng.normal(900, 250); failed = rng.integers(4, 12); freq = rng.normal(120, 35)
                duration = rng.normal(3, 1.2); src = rng.normal(18000, 5000); dst = rng.normal(2000, 700); ports = max(1, int(rng.normal(5, 2)))
            elif kind == "DoS":
                packet = rng.normal(1500, 350); failed = rng.integers(0, 4); freq = rng.normal(260, 60)
                duration = rng.normal(1.5, .7); src = rng.normal(70000, 18000); dst = rng.normal(1200, 500); ports = max(1, int(rng.normal(8, 3)))
            elif kind == "PortScan":
                packet = rng.normal(500, 160); failed = rng.integers(0, 4); freq = rng.normal(180, 45)
                duration = rng.normal(2.5, 1); src = rng.normal(12000, 3500); dst = rng.normal(2500, 800); ports = max(12, int(rng.normal(35, 8)))
            else:
                packet = rng.normal(1700, 400); failed = rng.integers(0, 3); freq = rng.normal(80, 25)
                duration = rng.normal(20, 7); src = rng.normal(90000, 25000); dst = rng.normal(40000, 12000); ports = max(3, int(rng.normal(10, 4)))

        rows.append([max(50, packet), max(0, failed), max(1, freq), max(.1, duration),
                     max(100, src), max(100, dst), ports, rng.choice(protocols), label, name])

    df = pd.DataFrame(rows, columns=[
        "packet_size","failed_logins","request_frequency","connection_duration",
        "source_bytes","destination_bytes","unique_ports","protocol","attack_label","label_name"
    ])
    DATA.parent.mkdir(exist_ok=True)
    df.to_csv(DATA, index=False)
    return df

if __name__ == "__main__":
    generate_dataset()
    print(f"Created {DATA}")
