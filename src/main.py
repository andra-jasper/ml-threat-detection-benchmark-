"""ML Threat Detection Benchmark.

Reproducible ETL, training, and evaluation for comparing threat-detection
models over identical data splits, with MLflow experiment tracking.

Pipeline:
    1. ``generate-data`` -- synthesize labeled telemetry (no external data).
    2. ``prepare``       -- clean, engineer features, stratified splits -> Parquet.
    3. ``benchmark``     -- train + evaluate every model, print comparison table.

Models (all CPU, all seeded):
    ensemble -- RandomForestClassifier (class-weighted)
    mlp      -- small PyTorch MLP classifier
    anomaly  -- IsolationForest trained on benign traffic only

Configuration via environment (all optional):
    MLFLOW_TRACKING_URI -- default "file:./mlruns"
    MODEL_DIR           -- default "./models"
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import mlflow

# MLflow >= 3 puts the local file-store tracking backend behind an opt-out
# flag; this benchmark intentionally uses a local file store.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_gen import LABELS, SEED, generate_synthetic_telemetry, write_csv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = ["ensemble", "mlp", "anomaly"]

FEATURE_COLS = [
    "duration_s",
    "bytes_sent",
    "bytes_received",
    "packets",
    "failed_logins",
    "dns_queries",
    "distinct_ports",
    "hour",
    "bytes_total",
    "byte_ratio",
    "pkts_per_sec",
]

LABEL_COL = "label"
SPLIT_COL = "split"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # of total; test gets the remaining 0.15


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")


def _model_dir() -> str:
    return os.environ.get("MODEL_DIR", "./models")


def _set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# ETL
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a cleaned telemetry frame."""
    df = df.copy()
    df["bytes_total"] = df["bytes_sent"] + df["bytes_received"]
    df["byte_ratio"] = df["bytes_sent"] / (df["bytes_received"] + 1.0)
    df["pkts_per_sec"] = df["packets"] / (df["duration_s"] + 1e-6)
    return df


def prepare(input_csv: str, output_parquet: str) -> None:
    """Clean, label, and stratify raw telemetry into train/val/test splits.

    Writes a single Parquet file to *output_parquet* with feature columns,
    the integer ``label`` column, and a ``split`` column
    ("train" / "val" / "test").
    """
    df = pd.read_csv(input_csv)

    # --- cleaning ------------------------------------------------------
    df = df.drop_duplicates().reset_index(drop=True)
    numeric = [c for c in FEATURE_COLS if c in df.columns]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].clip(lower=0)  # telemetry counters can't be negative
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype(int)
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

    # --- feature engineering -------------------------------------------
    df = _engineer_features(df)

    # --- stratified splits (class ratios preserved in every split) ------
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_FRAC, stratify=df[LABEL_COL], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df,
        train_size=VAL_FRAC / (1.0 - TRAIN_FRAC),
        stratify=temp_df[LABEL_COL],
        random_state=SEED,
    )
    train_df = train_df.copy(); val_df = val_df.copy(); test_df = test_df.copy()
    train_df[SPLIT_COL] = "train"
    val_df[SPLIT_COL] = "val"
    test_df[SPLIT_COL] = "test"

    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    parent = os.path.dirname(os.path.abspath(output_parquet))
    os.makedirs(parent, exist_ok=True)
    out.to_parquet(output_parquet, index=False)
    print(f"Wrote {len(out)} rows -> {output_parquet} "
          f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")


def _load_split(data_path: str, split: str):
    """Return (X, y) for one split of the processed Parquet file."""
    df = pd.read_parquet(data_path)
    sub = df[df[SPLIT_COL] == split].reset_index(drop=True)
    return sub[FEATURE_COLS].to_numpy(dtype=np.float64), sub[LABEL_COL].to_numpy(dtype=int)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _threat_scores(model_name: str, bundle: dict, X: np.ndarray) -> np.ndarray:
    """P(threat) / anomaly score for ROC-AUC: higher == more suspicious."""
    if model_name == "anomaly":
        return -bundle["model"].decision_function(X)  # higher == more anomalous
    proba = bundle["model"].predict_proba(X)
    return 1.0 - proba[:, 0]  # 1 - P(benign)


def _predict(model_name: str, bundle: dict, X: np.ndarray) -> np.ndarray:
    if model_name == "anomaly":
        raw = bundle["model"].predict(X)  # 1 normal, -1 anomaly
        return (raw == -1).astype(int)    # 1 == threat (any class)
    return bundle["model"].predict(X)


def _train_ensemble(X_train, y_train) -> dict:
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=2,
    )
    clf.fit(X_train, y_train)
    return {"model": clf, "params": {"n_estimators": 200, "class_weight": "balanced_subsample"}}


def _train_mlp(X_train, y_train) -> dict:
    _set_seeds(SEED)
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    n_classes = len(np.unique(y_train))
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    class_weight = torch.tensor(len(y_train) / (n_classes * np.maximum(counts, 1.0)),
                                dtype=torch.float32)

    net = _MLP(Xs.shape[1], n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    X_t = torch.tensor(Xs, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    net.train()
    epochs, batch = 20, 256
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        perm = torch.randperm(len(X_t), generator=g)
        for i in range(0, len(X_t), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net(X_t[idx]), y_t[idx])
            loss.backward()
            opt.step()

    # Attach a sklearn-style predict/predict_proba interface.
    class _Wrapped:
        def __init__(self, net, scaler):
            self.net, self.scaler = net, scaler

        def _proba(self, X):
            self.net.eval()
            with torch.no_grad():
                Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
                return torch.softmax(self.net(Xt), dim=1).numpy()

        def predict(self, X):
            return self._proba(X).argmax(axis=1)

        def predict_proba(self, X):
            return self._proba(X)

    return {
        "model": _Wrapped(net, scaler),
        "params": {"hidden": [64, 32], "epochs": epochs, "lr": 1e-3, "dropout": 0.2,
                   "n_classes": n_classes},
        "torch_state": net.state_dict(),
        "scaler": scaler,
    }


def _train_anomaly(X_train, y_train) -> dict:
    benign = X_train[y_train == 0]
    contamination = float((y_train != 0).mean())
    clf = IsolationForest(
        n_estimators=200,
        contamination=max(contamination, 0.01),
        random_state=SEED,
        n_jobs=2,
    )
    clf.fit(benign)
    return {"model": clf, "params": {"n_estimators": 200, "contamination": round(contamination, 4)}}


_TRAINERS = {"ensemble": _train_ensemble, "mlp": _train_mlp, "anomaly": _train_anomaly}


def _save_bundle(model_name: str, bundle: dict) -> str:
    os.makedirs(_model_dir(), exist_ok=True)
    if model_name == "mlp":
        path = os.path.join(_model_dir(), "mlp.pt")
        torch.save(
            {"state_dict": bundle["torch_state"],
             # stored as tensors so weights_only=True loading stays safe
             "scaler_mean": torch.tensor(bundle["scaler"].mean_),
             "scaler_scale": torch.tensor(bundle["scaler"].scale_),
             "in_dim": bundle["scaler"].mean_.shape[0],
             "n_classes": bundle["params"].get("n_classes", 4)},
            path,
        )
    else:
        path = os.path.join(_model_dir(), f"{model_name}.pkl")
        joblib.dump(bundle["model"], path)
    return path


def _load_bundle(model_name: str) -> dict:
    if model_name == "mlp":
        ckpt = torch.load(os.path.join(_model_dir(), "mlp.pt"), map_location="cpu", weights_only=True)
        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"].numpy()
        scaler.scale_ = ckpt["scaler_scale"].numpy()
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = ckpt["in_dim"]
        net = _MLP(ckpt["in_dim"], ckpt.get("n_classes", 4))
        net.load_state_dict(ckpt["state_dict"])

        class _Wrapped:
            def _proba(self, X):
                net.eval()
                with torch.no_grad():
                    Xt = torch.tensor(scaler.transform(X), dtype=torch.float32)
                    return torch.softmax(net(Xt), dim=1).numpy()

            def predict(self, X):
                return self._proba(X).argmax(axis=1)

            def predict_proba(self, X):
                return self._proba(X)

        return {"model": _Wrapped(), "params": {}}
    path = os.path.join(_model_dir(), f"{model_name}.pkl")
    return {"model": joblib.load(path), "params": {}}


# ---------------------------------------------------------------------------
# Training / evaluation (MLflow-tracked)
# ---------------------------------------------------------------------------

def _binary_threat_labels(y: np.ndarray) -> np.ndarray:
    return (y != 0).astype(int)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    """Multiclass where available; always includes binary threat-vs-benign."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    y_bin = _binary_threat_labels(y_true)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "roc_auc_threat": float(roc_auc_score(y_bin, scores)),
        "brier_threat": float(brier_score_loss(y_bin, np.clip(scores, 0, 1))),
        "per_class": {
            str(k): {"precision": float(v["precision"]), "recall": float(v["recall"]),
                     "f1": float(v["f1-score"]), "support": int(v["support"])}
            for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": cm.tolist(),
    }
    # Binary threat-vs-benign view (fair comparison incl. the anomaly model).
    y_pred_bin = _binary_threat_labels(y_pred)
    bin_report = classification_report(y_bin, y_pred_bin, output_dict=True, zero_division=0)
    metrics["threat_precision"] = float(bin_report["1"]["precision"])
    metrics["threat_recall"] = float(bin_report["1"]["recall"])
    metrics["threat_f1"] = float(bin_report["1"]["f1-score"])
    # Alias kept for the CLI table.
    metrics["f1"] = metrics["macro_f1"]
    return metrics


def train(model_name: str, data_path: str) -> dict:
    """Train *model_name* and return a metrics dict.

    Logs params, train metrics, and the model artifact to MLflow
    (local file store).
    """
    if model_name not in _TRAINERS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODELS}")
    X_train, y_train = _load_split(data_path, "train")

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment("ml-threat-detection-benchmark")
    with mlflow.start_run(run_name=f"{model_name}-train") as run:
        bundle = _TRAINERS[model_name](X_train, y_train)
        model_path = _save_bundle(model_name, bundle)

        mlflow.log_params({k: str(v) for k, v in
                           {"model": model_name, "seed": SEED,
                            "n_train": len(X_train), **bundle["params"]}.items()})
        y_pred = _predict(model_name, bundle, X_train)
        scores = _threat_scores(model_name, bundle, X_train)
        metrics = _compute_metrics(y_train, y_pred, scores)
        mlflow.log_metrics({k: v for k, v in metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.log_artifact(model_path)

        metrics["mlflow_run_id"] = run.info.run_id
        print(f"[{model_name}] train accuracy={metrics['accuracy']:.3f} "
              f"macro_f1={metrics['macro_f1']:.3f} run={run.info.run_id[:8]}")
        return metrics


def evaluate(model_name: str, data_path: str) -> dict:
    """Evaluate *model_name* on the held-out test split.

    Logs test metrics and the confusion matrix to MLflow; returns metrics.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODELS}")
    X_test, y_test = _load_split(data_path, "test")
    bundle = _load_bundle(model_name)

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment("ml-threat-detection-benchmark")
    with mlflow.start_run(run_name=f"{model_name}-eval") as run:
        y_pred = _predict(model_name, bundle, X_test)
        scores = _threat_scores(model_name, bundle, X_test)
        metrics = _compute_metrics(y_test, y_pred, scores)
        mlflow.log_params({"model": model_name, "seed": SEED, "n_test": len(X_test)})
        mlflow.log_metrics({k: v for k, v in metrics.items()
                            if isinstance(v, (int, float))})
        cm_path = os.path.join(_model_dir(), f"{model_name}_confusion_matrix.json")
        with open(cm_path, "w") as f:
            json.dump({"labels": LABELS, "matrix": metrics["confusion_matrix"]}, f, indent=2)
        mlflow.log_artifact(cm_path)
        metrics["mlflow_run_id"] = run.info.run_id
        return metrics


def benchmark(data_path: str = "./data/processed.parquet") -> None:
    """Train and evaluate every model, printing a comparison table."""
    results = {}
    for model_name in MODELS:
        train(model_name, data_path)
        results[model_name] = evaluate(model_name, data_path)

    header = f"{'model':<12}{'accuracy':>10}{'macro_f1':>10}{'weighted_f1':>12}{'roc_auc':>10}{'threat_f1':>11}"
    print("\nBenchmark comparison (test split)")
    print(header)
    print("-" * len(header))
    for model_name in MODELS:
        m = results[model_name]
        print(f"{model_name:<12}{m['accuracy']:>10.3f}{m['macro_f1']:>10.3f}"
              f"{m['weighted_f1']:>12.3f}{m['roc_auc_threat']:>10.3f}{m['threat_f1']:>11.3f}")


def _cmd_generate_data(args: list[str]) -> None:
    out = args[0] if args else "./data/telemetry.csv"
    n = int(args[1]) if len(args) > 1 else 5000
    df = generate_synthetic_telemetry(n_samples=n, seed=SEED)
    write_csv(df, out)
    print(f"Wrote {len(df)} rows to {out}")
    print("class balance:", df["label"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        data = sys.argv[2] if len(sys.argv) > 2 else "./data/processed.parquet"
        benchmark(data)
    elif len(sys.argv) > 1 and sys.argv[1] == "generate-data":
        _cmd_generate_data(sys.argv[2:])
    else:
        print("Usage:")
        print("  python src/main.py generate-data [output.csv] [n_samples]")
        print("  python src/main.py prepare <input.csv> <output.parquet>")
        print("  python src/main.py benchmark [processed.parquet]")
