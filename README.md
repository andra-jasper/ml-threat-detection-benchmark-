# ML Threat Detection Benchmark

A reproducible benchmark for comparing threat-detection models over identical
data splits, with MLflow experiment tracking. Everything runs end-to-end on
CPU with **no external data and no API keys** — the telemetry dataset is
synthesized in-repo with a fixed seed.

> Portfolio project demonstrating applied ML research practices: reproducible
> experiments, stratified evaluation on imbalanced data, and honest model
> comparison.

## What it does

1. **Synthetic telemetry** (`src/data_gen.py`) — generates labeled
   network-telemetry-like rows (duration, bytes, packets, failed logins, DNS
   queries, ports, hour) across 4 classes with realistic imbalance:
   ~88% benign, ~5% malware, ~4% intrusion, ~3% exfiltration. Seeded, so the
   same seed always yields the same CSV.
2. **ETL** — deduplicates, median-imputes missing values, engineers features
   (`bytes_total`, `byte_ratio`, `pkts_per_sec`), and writes stratified
   70/15/15 train/val/test splits to Parquet (class ratios preserved).
3. **Models** (all CPU, all seeded with `SEED=42`):
   - `ensemble` — `RandomForestClassifier` (200 trees, class-weighted)
   - `mlp` — small PyTorch MLP (11 → 64 → 32 → 4, dropout, class-weighted
     cross-entropy, 20 epochs)
   - `anomaly` — `IsolationForest` trained on benign traffic only
4. **MLflow tracking** — every train/eval run logs params, metrics, the model
   artifact, and the confusion matrix to a local file store (`./mlruns`).
5. **Evaluation** — per-class precision/recall/F1, ROC-AUC (threat vs benign),
   Brier score, and confusion matrix; `benchmark` prints a comparison table.

Note: the original skeleton listed BERT as a candidate model. BERT was
deliberately replaced with a small PyTorch MLP: this benchmark is designed to
run fully offline with zero downloads, and a from-scratch MLP keeps the
comparison honest on the synthetic tabular data.

## Quickstart

```bash
# 1. Enter the project and install dependencies
cd ml-threat-detection-benchmark-
pip install -r requirements.txt

# 2. Generate synthetic telemetry (no external data needed)
python src/main.py generate-data ./data/telemetry.csv 5000

# 3. Run ETL -> processed Parquet with stratified splits
python src/main.py prepare ./data/telemetry.csv ./data/processed.parquet

# 4. Train + evaluate all models, print the comparison table
python src/main.py benchmark ./data/processed.parquet

# 5. Browse runs, params, metrics, and artifacts
mlflow ui
```

Sample output of step 4:

```
Benchmark comparison (test split)
model         accuracy  macro_f1 weighted_f1   roc_auc  threat_f1
-----------------------------------------------------------------
ensemble         0.999     0.996       0.999     1.000      0.994
mlp              1.000     1.000       1.000     1.000      1.000
anomaly          0.820     0.321       0.840     0.970      0.664
```

(The anomaly detector ranks threats well — ROC-AUC 0.97 — but, having never
seen labeled threats, classifies them less precisely. That trade-off is the
point of the comparison.)

## Reading MLflow results

```bash
mlflow ui  # then open http://127.0.0.1:5000
```

- Experiment: **ml-threat-detection-benchmark**
- Runs are named `<model>-train` / `<model>-eval`.
- Each run carries hyperparameters, train/test metrics, the serialized model,
  and (for eval runs) the confusion matrix JSON.

Override the local stores with environment variables:

```bash
MLFLOW_TRACKING_URI=file:/path/to/mlruns MODEL_DIR=/path/to/models \
  python src/main.py benchmark ./data/processed.parquet
```

## Running tests

```bash
pytest tests/ -v
```

The suite (13 tests) covers: ETL output shapes, split ratios, stratification,
determinism (same seed → identical output), per-model training metrics,
evaluation metrics, MLflow run logging, and the benchmark table output. Tests
run against a small seeded dataset in an isolated tmp dir — no network, no
external data.

## Project structure

```
ml-threat-detection-benchmark-/
├── src/
│   ├── __init__.py
│   ├── data_gen.py   # seeded synthetic telemetry generator
│   └── main.py       # ETL + training + evaluation + CLI
├── tests/
│   ├── conftest.py
│   └── test_benchmark.py
├── requirements.txt
└── README.md
```

## Reproducibility

- Single seed constant (`SEED = 42`) threads through data generation,
  splitting (`random_state`), model init, and PyTorch minibatch shuffling.
- `tests/test_benchmark.py::test_etl_is_deterministic` asserts byte-identical
  ETL output across runs.
