"""ML Threat Detection Benchmark.

Portfolio skeleton: reproducible ETL, training, and evaluation for
comparing threat-detection models (BERT, ensembles, anomaly detectors)
with MLflow experiment tracking.

Extend the marked TODO sections with your own dataset and model
configurations.
"""

import sys

MODELS = ["bert", "ensemble", "anomaly"]


def prepare(input_csv: str, output_parquet: str) -> None:
    """Clean, label, and stratify raw telemetry into train/val/test splits.

    Writes a processed Parquet file to *output_parquet*.
    """
    # TODO: implement Pandas/NumPy ETL: cleaning, labeling,
    # stratified splitting, and Parquet export.
    raise NotImplementedError("ETL pipeline not yet implemented")


def train(model_name: str, data_path: str) -> dict:
    """Train *model_name* and return a metrics dict.

    Logs params, metrics, and artifacts to MLflow.
    """
    # TODO: implement training per model with fixed seeds and
    # mlflow.start_run() logging.
    raise NotImplementedError(f"Training not yet implemented for {model_name}")


def evaluate(model_name: str, data_path: str) -> dict:
    """Evaluate *model_name*: per-class precision/recall/F1 + calibration."""
    # TODO: implement stratified evaluation and error analysis.
    raise NotImplementedError(f"Evaluation not yet implemented for {model_name}")


def benchmark(data_path: str = "./data/processed.parquet") -> None:
    """Train and evaluate every model, printing a comparison table."""
    results = {}
    for model_name in MODELS:
        train(model_name, data_path)
        results[model_name] = evaluate(model_name, data_path)
    print(f"{'model':<12}{'f1':>8}")
    for model_name, metrics in results.items():
        print(f"{model_name:<12}{metrics.get('f1', float('nan')):>8.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark()
    else:
        print("Usage:")
        print('  python src/main.py prepare <input.csv> <output.parquet>')
        print("  python src/main.py benchmark")
