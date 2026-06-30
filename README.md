# CASCADE: Real-Time Click-Through-Rate Prediction Pipeline

CASCADE is an end-to-end Machine Learning pipeline designed to predict Click-Through Rates (CTR) in real-time. It explicitly addresses the extreme class imbalance problem inherent in CTR datasets (often ~1-2% positive rate) and is optimized for low-latency inference in production.

## Features
- **Data Engineering**: Robust handling of categorical features using target encoding with smoothing for cold-start problems.
- **Class Imbalance**: Dynamic `scale_pos_weight` calibration for Gradient Boosting models to properly learn rare click events.
- **Hyperparameter Tuning**: Fully automated tuning using Optuna.
- **MLOps**: MLflow integration for tracking experiments and saving model artifacts.
- **Serving**: Ultra-fast FastAPI backend optimized for sub-millisecond latency.
- **Containerization**: Docker and Docker Compose ready for production deployments.

## Project Structure
```
CASCADE/
├── data/              # Data generation and robust loading
├── features/          # Target encoding and temporal feature engineering
├── imbalance/         # Class imbalance logic
├── models/            # LightGBM, XGBoost, and Baseline wrappers
├── eval/              # Comprehensive metrics (AUC, PR-AUC, LogLoss)
├── tune/              # Optuna hyperparameter optimization scripts
├── serve/             # FastAPI deployment code
├── docker-compose.yml # Container orchestration
└── config.yaml        # Centralized project configuration
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate synthetic data** (for testing):
   ```bash
   python data/generate_sample.py
   ```
3. **Run Hyperparameter Tuning & Training**:
   ```bash
   python tune/optuna_search.py
   ```
   *View experiments by running `mlflow ui --backend-store-uri sqlite:///mlflow.db`*
4. **Serve Model**:
   ```bash
   python serve/api.py
   ```
   *Test the API at http://localhost:8000/docs*

## Docker Deployment
```bash
docker-compose up --build
```
