from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.lightgbm
import pandas as pd
import time

app = FastAPI(title="Cascade CTR Prediction API", description="Real-time CTR prediction serving engine")

# Global variables for model and encoders
# In a real production system, we'd load these from a model registry like MLflow
MODEL = None

class PredictionRequest(BaseModel):
    user_id: str
    ad_id: str
    campaign_id: str
    category: str
    device_type: str
    age_group: str
    hour: int
    day_of_week: int
    position: int

@app.on_event("startup")
def load_model():
    global MODEL
    try:
        # Load the latest model from MLflow
        # Note: In a real scenario you would point this to the correct run_id
        # For this skeleton, we assume the model is logged as "best_lgbm_model" 
        # but in practice, you might pass the path via env variables.
        # This is a placeholder showing the intent.
        pass
    except Exception as e:
        print(f"Warning: Model could not be loaded on startup: {e}")

@app.post("/predict")
def predict_ctr(request: PredictionRequest):
    start_time = time.time()
    
    # 1. Feature Engineering (would use cached values in production)
    # 2. Encoding
    # 3. Prediction
    
    # Dummy prediction to demonstrate API structure and latency
    dummy_prob = 0.025
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "user_id": request.user_id,
        "ad_id": request.ad_id,
        "click_probability": dummy_prob,
        "latency_ms": round(latency_ms, 2)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    host = config.get('serve', {}).get('host', '0.0.0.0')
    port = config.get('serve', {}).get('port', 8000)
    
    uvicorn.run("api:app", host=host, port=port, reload=True)
