import yaml
import joblib
import os
import sys

# Ensure parent directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.load import load_data_and_split
from features.engineer import FeatureEngineer
from features.encode import CategoricalEncoder
from models.gbm import get_lightgbm_model

def main():
    print("Loading data...")
    train, _ = load_data_and_split()
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("Fitting FeatureEngineer...")
    fe = FeatureEngineer(config)
    train_transformed = fe.fit(train).transform(train)
    
    print("Encoding categoricals...")
    ce = CategoricalEncoder(config)
    
    # Keep only features specified in config + generated features + target
    expected_cols = config['features']['user_cols'] + config['features']['ad_cols'] + config['features']['context_cols'] + ['click']
    if 'cross_cols' in config['features']:
        expected_cols += [f"{c[0]}_x_{c[1]}" for c in config['features']['cross_cols']]
    expected_cols += ['hour_of_day', 'day_of_week', 'is_weekend', 'is_night']
    expected_cols = [c for c in expected_cols if c != 'hour']
    
    train_transformed = train_transformed[[c for c in expected_cols if c in train_transformed.columns]]
    train_X = ce.fit(train_transformed, train_transformed['click']).transform(train_transformed)
    train_y = train_X.pop('click')
    
    print("Training best LightGBM model for inference...")
    # Best params from recent Optuna run, but remove scale_pos_weight so probabilities are realistic!
    best_params = {
        'n_estimators': 123, 
        'learning_rate': 0.06158318932815055, 
        'max_depth': 8, 
        'num_leaves': 22, 
        'objective': 'binary'
    }
    model = get_lightgbm_model(best_params)
    model.fit(train_X, train_y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(fe, 'models/fe.pkl')
    joblib.dump(ce, 'models/ce.pkl')
    joblib.dump(model, 'models/model.pkl')
    print("Successfully saved fe.pkl, ce.pkl, and model.pkl in models/ directory.")

if __name__ == '__main__':
    main()
