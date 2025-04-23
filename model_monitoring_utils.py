import mlflow
import pandas as pd
import os
from typing import List, Dict

def get_latest_runs(experiment_name: str, n: int = 5) -> pd.DataFrame:
    """
    Fetch the latest n runs for a given MLflow experiment.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=n
    )
    if not runs:
        return pd.DataFrame()
    df = pd.DataFrame([run.data.metrics for run in runs])
    df['run_id'] = [run.info.run_id for run in runs]
    df['start_time'] = [run.info.start_time for run in runs]
    return df

def get_run_params(run_id: str) -> Dict:
    """
    Fetch parameters for a specific MLflow run.
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.params

def get_run_metrics(run_id: str) -> Dict:
    """
    Fetch metrics for a specific MLflow run.
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics

# Optionally: add more utility functions for drift detection, etc.
