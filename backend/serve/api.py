from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd
import time

app = FastAPI(title="Cascade CTR Prediction API", description="Real-time CTR prediction serving engine")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and encoders
MODEL = None
FE = None
CE = None

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
    global MODEL, FE, CE
    import joblib
    import os
    
    try:
        print("Loading Model and Encoders from disk...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        FE = joblib.load(os.path.join(base_dir, 'models', 'fe.pkl'))
        CE = joblib.load(os.path.join(base_dir, 'models', 'ce.pkl'))
        MODEL = joblib.load(os.path.join(base_dir, 'models', 'model.pkl'))
        print("Successfully loaded model and encoders.")
    except Exception as e:
        print(f"Warning: Model could not be loaded on startup: {e}")

@app.post("/predict")
def predict_ctr(request: PredictionRequest):
    start_time = time.time()
    
    if MODEL is None or FE is None or CE is None:
        raise HTTPException(status_code=500, detail="Model or Encoders not loaded.")
        
    # Map UI strings to actual Avazu encoded values for authenticity
    device_map = {"mobile": "1541007", "desktop": "1541008", "tablet": "1541009"}
    cat_map = {"electronics": "7931", "fashion": "7932", "home": "7938", "sports": "7943"}
    pos_map = {1: "17", 2: "18", 3: "23", 4: "19", 5: "25", 6: "22", 7: "24", 8: "20", 9: "30", 10: "27"}
    
    device_val = device_map.get(request.device_type.lower(), "1541007")
    cat_val = cat_map.get(request.category.lower(), "7931")
    pos_val = pos_map.get(request.position, "17")
    
    # 1. Feature Engineering
    # Map frontend request schema to Avazu features
    data = {
        "device_type": [device_val],
        "site_category": [cat_val],
        "banner_pos": [pos_val],
        "hour": [request.hour],
        "day_of_week": [request.day_of_week]
    }
    df = pd.DataFrame(data)
    
    # 1. Feature Engineering
    df = FE.transform(df)
    
    # 2. Encoding
    df = CE.transform(df)
    
    # 3. Drop unused columns
    cols_to_drop = ['hour']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # ENSURE exact column order!
    df = df[MODEL.feature_name_]
    print("Inference DF:", df.to_dict('records'))
    
    # 4. Prediction
    prob = MODEL.predict_proba(df)[0, 1]
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "user_id": request.user_id,
        "ad_id": request.ad_id,
        "click_probability": float(prob),
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
