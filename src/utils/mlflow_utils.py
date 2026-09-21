"""
src/utils/mlflow_utils.py
==========================
MLflow experiment tracking utilities.

CONCEPT: What is Experiment Tracking?
---------------------------------------
When you train a ML model, you make many choices:
  - Which model architecture? (yolo11n vs yolo11s)
  - How many epochs? (50 vs 100)
  - What learning rate? (0.001 vs 0.01)
  - What batch size? (16 vs 32)

Each combination gives a different result. How do you remember which
combination gave the best result?

Without tracking: you write notes in a text file, lose them, can't
compare runs side by side, can't reproduce old results.

With MLflow:
  - Every training run is logged automatically
  - All hyperparameters + metrics are stored in a database
  - You can compare runs in a web UI with charts
  - You can load any past model for inference

MLflow stores:
  RUNS
  ├── Parameters: {epochs: 50, batch: 16, model: yolo11n, ...}
  ├── Metrics: {mAP50: 0.891, loss: 0.24, ...} (logged per epoch)
  ├── Artifacts: {best.pt, confusion_matrix.png, training_curves.png}
  └── Tags: {phase: "3", dataset: "hagrid_subset"}

CONCEPT: The MLflow UI
-----------------------
After any training run, you can view results at:
    mlflow ui --port 5000
Then open: http://localhost:5000

You'll see a table of all runs with their metrics, hyperparameters,
and a chart comparing all runs. This is standard practice at ML companies.
"""

from __future__ import annotations

from pathlib import Path


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get an existing MLflow experiment or create it if it doesn't exist.

    CONCEPT: MLflow Experiments
    ----------------------------
    An "experiment" is a group of related runs.
    We use one experiment per project phase:
      - "hand-detection-yolo"     ← Phase 3
      - "sign-classifier"         ← Phase 5

    Args:
        experiment_name: Name of the experiment

    Returns:
        Experiment ID string
    """
    import mlflow

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"project": "hand-sign-detection", "version": "1.0"},
        )
        print(f"✅ Created MLflow experiment: '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"📂 Using existing MLflow experiment: '{experiment_name}' (id={experiment_id})")

    return experiment_id


def log_yolo_training_run(
    run_name: str,
    hyperparams: dict,
    metrics: dict,
    artifacts_dir: str | Path | None = None,
    experiment_name: str = "hand-detection-yolo",
) -> str:
    """
    Log a complete YOLO training run to MLflow.

    Args:
        run_name:       Human-readable name for this run
        hyperparams:    Dict of training hyperparameters
        metrics:        Dict of evaluation metrics
        artifacts_dir:  Directory with model weights and plots to save
        experiment_name: MLflow experiment to log to

    Returns:
        MLflow run ID
    """
    import mlflow

    experiment_id = get_or_create_experiment(experiment_name)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
    ) as run:
        # CONCEPT: Logging parameters
        # Parameters are fixed values set BEFORE training (hyperparameters).
        # They describe HOW the model was trained.
        mlflow.log_params(hyperparams)

        # CONCEPT: Logging metrics
        # Metrics are values measured AFTER or DURING training.
        # They describe HOW WELL the model performs.
        mlflow.log_metrics(metrics)

        # CONCEPT: Logging artifacts
        # Artifacts are files: model weights, plots, configs.
        # MLflow stores them so you can reload any past model.
        if artifacts_dir:
            artifacts_dir = Path(artifacts_dir)
            # Log model weights
            for pt_file in artifacts_dir.rglob("*.pt"):
                mlflow.log_artifact(str(pt_file), artifact_path="weights")
            # Log training plots
            for png_file in artifacts_dir.rglob("*.png"):
                mlflow.log_artifact(str(png_file), artifact_path="plots")
            # Log training results CSV
            for csv_file in artifacts_dir.rglob("results.csv"):
                mlflow.log_artifact(str(csv_file), artifact_path="results")

        run_id = run.info.run_id
        print("\n📊 MLflow run logged:")
        print(f"   Run ID:  {run_id}")
        print(f"   Name:    {run_name}")
        print(f"   mAP@0.5: {metrics.get('mAP50', 'N/A')}")
        print("\n   View at: mlflow ui --port 5000 → http://localhost:5000")

    return run_id


def log_epoch_metrics(
    epoch: int,
    metrics: dict,
    run_id: str | None = None,
) -> None:
    """
    Log per-epoch metrics during training (loss, mAP per epoch).

    CONCEPT: Step-wise logging
    ---------------------------
    MLflow can log metrics as a TIME SERIES (one value per epoch).
    This lets you see the training curve:
      epoch 1: mAP=0.12
      epoch 5: mAP=0.45
      epoch 20: mAP=0.87  ← converged

    If you see the curve plateauing early, training can stop.
    If it's still improving at epoch 50, maybe train for longer.
    """
    import mlflow

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics, step=epoch)
