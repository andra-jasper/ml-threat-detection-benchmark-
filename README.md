# ML Threat Detection Benchmark

A reproducible benchmark for comparing threat-detection models — BERT, ensemble methods, and anomaly detectors — over consistent data processing and evaluation steps, with MLflow experiment tracking.

> Portfolio project demonstrating applied ML research practices: reproducible experiments, stratified evaluation, and error analysis.

## Features

- **Consistent pipeline** — shared ETL for cleaning and labeling security telemetry (Pandas/NumPy)
- **Model comparison** — BERT, scikit-learn ensembles, and anomaly detectors on identical splits
- **Experiment tracking** — every run logged to MLflow with params, metrics, and artifacts
- **Stratified evaluation** — precision/recall/F1 plus error analysis by threat class

## Tech Stack

`Python` `PyTorch` `scikit-learn` `MLflow` `Pandas` `NumPy`

## Quickstart

```bash
# 1. Clone and enter the project
git clone https://github.com/andra-jasper/ml-threat-detection-benchmark-.git
cd ml-threat-detection-benchmark-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data (expects a raw telemetry CSV)
python src/main.py prepare ./data/telemetry.csv ./data/processed.parquet

# 4. Train + evaluate all models
python src/main.py benchmark

# 5. View results
mlflow ui
```

## Project Structure

```
ml-threat-detection-benchmark-/
├── src/
│   ├── __init__.py
│   └── main.py          # ETL + training + evaluation entrypoint (skeleton)
├── requirements.txt
└── README.md
```

## Methodology

1. **ETL** — clean, label, and stratify telemetry into train/val/test splits.
2. **Train** — fit each model with fixed seeds and logged hyperparameters.
3. **Evaluate** — report per-class precision/recall/F1 and calibration.
4. **Analyze** — error analysis surfacing where detectors disagree.

## Note

This is a portfolio skeleton showing research-engineering practices. Bring your own labeled dataset to run the full benchmark.
