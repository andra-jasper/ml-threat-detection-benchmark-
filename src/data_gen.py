"""Synthetic network-telemetry generator for the threat-detection benchmark.

Produces labeled, network-telemetry-like rows with realistic class
imbalance (threats are rare). Everything is seeded, so the same
``(n_samples, seed)`` pair always yields the same CSV.

Label encoding:
    0 = benign        (~88%)
    1 = malware       (~5%)   -- beaconing / C2-like DNS + port behaviour
    2 = intrusion     (~4%)   -- brute-force-like failed logins
    3 = exfiltration  (~3%)   -- large outbound transfers
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEED = 42

LABELS = ["benign", "malware", "intrusion", "exfiltration"]

COLUMNS = [
    "duration_s",
    "bytes_sent",
    "bytes_received",
    "packets",
    "failed_logins",
    "dns_queries",
    "distinct_ports",
    "hour",
    "label",
]


def generate_synthetic_telemetry(n_samples: int = 5000, seed: int = SEED) -> pd.DataFrame:
    """Generate *n_samples* labeled telemetry rows.

    Each threat class is drawn from a shifted distribution so the classes
    are learnable but overlapping (non-trivial). A small fraction of NaNs
    and duplicate rows is injected to exercise the ETL cleaning step.
    """
    rng = np.random.default_rng(seed)

    # Class priors: threats are rare, as in real telemetry.
    priors = np.array([0.88, 0.05, 0.04, 0.03])
    labels = rng.choice(len(LABELS), size=n_samples, p=priors)

    n = n_samples
    duration_s = np.abs(rng.normal(30.0, 25.0, n)).clip(min=0.5)
    bytes_sent = np.abs(rng.normal(5_000, 4_000, n))
    bytes_received = np.abs(rng.normal(20_000, 15_000, n))
    packets = np.abs(rng.normal(120, 90, n)).clip(min=1).astype(int)
    failed_logins = rng.poisson(0.2, n)
    dns_queries = rng.poisson(3, n)
    distinct_ports = rng.poisson(2, n) + 1
    hour = rng.integers(0, 24, n)

    # --- threat-class shifts -------------------------------------------
    is_malware = labels == 1
    dns_queries = np.where(is_malware, rng.poisson(25, n), dns_queries)
    distinct_ports = np.where(is_malware, rng.poisson(12, n) + 1, distinct_ports)

    is_intrusion = labels == 2
    failed_logins = np.where(is_intrusion, rng.poisson(15, n), failed_logins)
    duration_s = np.where(is_intrusion, np.abs(rng.normal(8.0, 6.0, n)).clip(min=0.5), duration_s)

    is_exfil = labels == 3
    bytes_sent = np.where(is_exfil, np.abs(rng.normal(500_000, 200_000, n)), bytes_sent)
    packets = np.where(is_exfil, np.abs(rng.normal(4_000, 1_500, n)).clip(min=1).astype(int), packets)

    df = pd.DataFrame(
        {
            "duration_s": duration_s,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "packets": packets,
            "failed_logins": failed_logins,
            "dns_queries": dns_queries,
            "distinct_ports": distinct_ports,
            "hour": hour,
            "label": labels,
        }
    )

    # Inject data-quality issues for the ETL step to clean up.
    n_nan = max(1, int(0.02 * n))
    nan_idx = rng.choice(n, size=n_nan, replace=False)
    df.loc[nan_idx, "bytes_received"] = np.nan
    dup_idx = rng.choice(n, size=max(1, int(0.01 * n)), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    return df[COLUMNS]


def write_csv(df: pd.DataFrame, path: str) -> str:
    """Write *df* to *path* (creating parent dirs) and return the path."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "./data/telemetry.csv"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    df = generate_synthetic_telemetry(n_samples=n, seed=SEED)
    write_csv(df, out)
    print(f"Wrote {len(df)} rows to {out}")
    print(df["label"].value_counts(normalize=True).round(3).to_dict())
