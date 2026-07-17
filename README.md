# CASCADE — Real-Time Click-Through Rate Prediction Engine
An intelligent, full-stack Machine Learning application designed to predict the likelihood of a user clicking on a specific advertisement in real-time with sub-50ms inference latency.

Built to solve real-world advertising queries by combining robust Feature Engineering, Target Encoding, and Gradient Boosting models, all served through a lightning-fast API and a premium React dashboard.

🎬 Demo Videos
*(Provide a link to your Loom or Demo Video here!)*

| Video | Description |
| :--- | :--- |
| **CASCADE PIPELINE & UI DEMO** | A complete walkthrough of the project — showing the Deep Space & Neon React UI, the live sub-50ms predictions, the FastAPI backend translation layer, and the MLflow hyperparameter tuning registry. |

## 🎯 Project Goal
The primary objective of this project is to build an authentic, production-ready AI system capable of predicting Click-Through Rates (CTR) based on real-world advertising data. It handles:

1. **High-Cardinality Features**: Using advanced Target Encoding to convert millions of unique categorical IDs (like device types and ad positions) into mathematical signals.
2. **Cold-Start & Default Fallbacks**: Intelligently handling unknown inputs from the user interface and mapping them to global dataset averages.
3. **Imbalanced Data Learning**: Handling the extreme class imbalance inherent in CTR data (where 98% of ads are never clicked) using dynamic scale weights in LightGBM.

The system relies on the authentic Avazu CTR dataset from Kaggle, ensuring all predictions are grounded in real historical interactions.

## 🏗️ System Architecture & High-Level Design

CASCADE relies on a modern Monorepo Architecture, separating concerns between the ML Pipeline, the API layer, and the Frontend UI:

- **Machine Learning Pipeline (Python/LightGBM)**: 
  - Downloads and parses real-world dataset chunks via HuggingFace.
  - Generates cross-features (e.g., `device_type_x_site_category`) and temporal features (`hour_of_day`, `is_night`).
  - Trains a LightGBM classifier with hyperparameter optimization powered by **Optuna**.
  - Logs all metrics, runs, and artifacts to a local **MLflow** registry.
- **Serving Engine (FastAPI)**: 
  - Loads the pre-trained LightGBM model and Category Encoders into memory.
  - Translates human-readable strings from the frontend (e.g., "Mobile") into authentic Kaggle dataset IDs.
  - Exposes a low-latency `/predict` endpoint.
- **Interactive Dashboard (React/Vite)**: 
  - A beautiful, responsive "Deep Space & Neon" interface built with Tailwind CSS.
  - Sends user interaction data to the API and dynamically visualizes the returning probability via an animated gauge.

### Bonus Features Implemented
✅ **End-to-End Deployment Ready**: Includes a `render.yaml` Infrastructure-as-Code file and a lightweight `requirements-serve.txt` for easy, free cloud hosting on Vercel and Render.
✅ **Authentic Data Mapping**: Instead of using synthetic noise, the backend translates clean UI inputs into genuine, anonymized hashed integers from the Avazu dataset.
✅ **Interactive UI Micro-Animations**: Built with a premium aesthetic featuring glassmorphism and subtle glowing interactions.

## 💻 Tech Stack
**Machine Learning & Backend**
- Framework: Python, FastAPI
- ML Models: LightGBM, Category Encoders
- Hyperparameter Tuning: Optuna
- MLOps: MLflow
- Data Processing: Pandas, NumPy

**Frontend**
- Framework: React (Vite)
- Styling: Tailwind CSS
- Icons: Lucide React
- HTTP Client: Axios

## 📂 Codebase & Folder Structure
The repository is highly organized for code clarity, separating the backend ML logic from the frontend UI.

```text
CASCADE/
├── backend/                  # Python Machine Learning & API
│   ├── data/                 # Dataset loaders and splitters
│   ├── features/             # Target Encoders and Feature Engineering
│   ├── models/               # Saved model artifacts (.pkl)
│   ├── serve/                # FastAPI backend engine (api.py)
│   ├── tune/                 # Optuna hyperparameter tuning (optuna_search.py)
│   ├── requirements.txt      # Training dependencies
│   ├── requirements-serve.txt# Lightweight serving dependencies
│   └── mlflow.db             # MLflow local tracking database
├── frontend/                 # React UI
│   ├── src/                  # React components and App.jsx
│   ├── public/               # Static assets
│   ├── package.json          # Node dependencies
│   └── tailwind.config.js    # Tailwind styling config
├── render.yaml               # Deployment config for Render
└── README.md                 # You are here
```

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.10+
- Node.js & npm

### 2. Backend & MLflow Setup
Open a terminal and navigate to the backend directory:
```bash
cd backend
python -m venv venv
```
Activate the virtual environment:
- **Windows**: `.\venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Servers (3 Terminals Required)

**Terminal 1: Start the FastAPI Backend**
```bash
cd backend
.\venv\Scripts\activate
uvicorn serve.api:app --reload
```
*(The backend runs on http://localhost:8000)*

**Terminal 2: Start MLflow Dashboard**
```bash
cd backend
.\venv\Scripts\activate
mlflow ui
```
*(View your model training history at http://localhost:5000)*

**Terminal 3: Start the React Frontend**
```bash
cd frontend
npm install
npm run dev
```
*(View the beautiful Deep Space dashboard at http://localhost:5173)*

## 🌍 Deployment
CASCADE is configured for instant, free deployment:

- **Backend**: Connect your GitHub to **Render.com**. It will automatically detect the `render.yaml` file in the root directory and deploy the FastAPI server using the lightweight `requirements-serve.txt`.
- **Frontend**: Connect your GitHub to **Vercel**. Add the `VITE_API_URL` environment variable pointing to your deployed Render URL.

---
*© Copyright 2026. Made with love.*
