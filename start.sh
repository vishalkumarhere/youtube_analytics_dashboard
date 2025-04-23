#!/bin/bash
# Start MLflow tracking server in background
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --host 0.0.0.0 --port 5000 &

# Start Streamlit app
streamlit run app.py
