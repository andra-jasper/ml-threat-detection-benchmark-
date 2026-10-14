"""End-to-end tests for the ML threat-detection benchmark.

All tests run against a small seeded synthetic dataset in an isolated
tmp dir (own MLflow file store + model dir), so they are fast and
deterministic.
"""

import math

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

import main
from data_gen import SEED, generate_synthetic_telemetry, write_csv

N_TEST_ROWS = 900


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """Isolated run: fresh CSV -> prepared parquet, own mlruns + models."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))

    raw_csv = tmp_path / "telemetry.csv"
    write_csv(generate_synthetic_telemetry(n_samples=N_TEST_ROWS, seed=SEED), str(raw_csv))
    processed = tmp_path / "processed.parquet"
    main.prepare(str(raw_csv), str(processed))
    return {"csv": str(raw_csv), "parquet": str(processed), "root": tmp_path}


def _frame(workdir):
    return pd.read_parquet(workdir["parquet"])


# ---------------------------------------------------------------------------
# ETL
# ---------------------------------------------------------------------------

def test_etl_output_shape_and_columns(workdir):
    df = _frame(workdir)
    assert set(main.FEATURE_COLS) <= set(df.columns)
    assert main.LABEL_COL in df.columns
    assert main.SPLIT_COL in df.columns
    assert df[main.FEATURE_COLS].isna().sum().sum() == 0  # imputed
    assert (df[main.FEATURE_COLS] >= 0).all().all()        # clipped


def test_etl_split_ratios(workdir):
    df = _frame(workdir)
    frac = df[main.SPLIT_COL].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.03)
    assert frac["val"] == pytest.approx(0.15, abs=0.03)
    assert frac["test"] == pytest.approx(0.15, abs=0.03)


def test_etl_splits_are_stratified(workdir):
    """Class proportions are preserved across splits (stratification)."""
    df = _frame(workdir)
    overall = df[main.LABEL_COL].value_counts(normalize=True).sort_index()
    for split in ("train", "val", "test"):
        sub = df[df[main.SPLIT_COL] == split][main.LABEL_COL].value_counts(normalize=True).sort_index()
        for cls in overall.index:
            assert sub.get(cls, 0.0) == pytest.approx(overall[cls], abs=0.05)


def test_etl_is_deterministic(workdir, tmp_path):
    """Same seed -> identical processed output."""
    second = tmp_path / "processed2.parquet"
    main.prepare(workdir["csv"], str(second))
    pd.testing.assert_frame_equal(_frame(workdir), pd.read_parquet(second))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", main.MODELS)
def test_each_model_trains_and_returns_metrics(workdir, model_name):
    metrics = main.train(model_name, workdir["parquet"])
    for key in ("accuracy", "macro_f1", "weighted_f1", "roc_auc_threat",
                "threat_f1", "confusion_matrix", "mlflow_run_id", "per_class"):
        assert key in metrics, f"missing {key} for {model_name}"
    for key in ("accuracy", "macro_f1", "weighted_f1", "roc_auc_threat", "threat_f1"):
        assert 0.0 <= metrics[key] <= 1.0 and math.isfinite(metrics[key])
    assert metrics["accuracy"] > 0.5  # synthetic classes are separable


@pytest.mark.parametrize("model_name", main.MODELS)
def test_evaluate_returns_test_metrics(workdir, model_name):
    main.train(model_name, workdir["parquet"])
    metrics = main.evaluate(model_name, workdir["parquet"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "mlflow_run_id" in metrics
    assert len(metrics["confusion_matrix"]) >= 2  # >=2x2 matrix


def test_unknown_model_raises(workdir):
    with pytest.raises(ValueError):
        main.train("bert", workdir["parquet"])


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def test_mlflow_runs_logged(workdir):
    for model_name in main.MODELS:
        main.train(model_name, workdir["parquet"])
    client = MlflowClient(tracking_uri=f"file:{workdir['root'] / 'mlruns'}")
    exp = client.get_experiment_by_name("ml-threat-detection-benchmark")
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) >= len(main.MODELS)
    run = runs[0]
    assert "model" in run.data.params
    assert "accuracy" in run.data.metrics
    assert len(run.data.tags) >= 0


# ---------------------------------------------------------------------------
# Benchmark CLI
# ---------------------------------------------------------------------------

def test_benchmark_prints_comparison_table(workdir, capsys):
    main.benchmark(workdir["parquet"])
    out = capsys.readouterr().out
    assert "Benchmark comparison" in out
    assert "accuracy" in out and "macro_f1" in out and "roc_auc" in out
    for model_name in main.MODELS:
        assert model_name in out
