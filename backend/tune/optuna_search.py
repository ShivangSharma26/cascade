import optuna
import mlflow
import yaml
import sys
import os

# Add parent dir to path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load import load_data_and_split
from features.engineer import FeatureEngineer
from features.encode import CategoricalEncoder
from imbalance.handle import calculate_class_weights
from models.gbm import get_lightgbm_model
from eval.evaluate import evaluate_model, print_evaluation

def objective(trial, config, train_X, train_y, test_X, test_y, weight):
    # Hyperparameters to tune
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'scale_pos_weight': weight,
        'random_state': config['data']['random_state'],
        'verbose': -1
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        model = get_lightgbm_model(params)
        model.fit(train_X, train_y)
        
        preds = model.predict_proba(test_X)[:, 1]
        results = evaluate_model(test_y, preds)
        
        mlflow.log_metrics(results)
        
        # We optimize for PR-AUC because of extreme imbalance
        return results['pr_auc']

def run_pipeline():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("Loading data...")
    train, test = load_data_and_split()
    
    print("Engineering features...")
    fe = FeatureEngineer(config)
    train = fe.fit(train).transform(train)
    test = fe.transform(test)
    
    print("Encoding categoricals...")
    ce = CategoricalEncoder(config)
    
    # Keep only features specified in config + generated features + target
    expected_cols = config['features']['user_cols'] + config['features']['ad_cols'] + config['features']['context_cols'] + ['click']
    if 'cross_cols' in config['features']:
        expected_cols += [f"{c[0]}_x_{c[1]}" for c in config['features']['cross_cols']]
    expected_cols += ['hour_of_day', 'day_of_week', 'is_weekend', 'is_night']
    expected_cols = [c for c in expected_cols if c != 'hour']
    
    train = train[[c for c in expected_cols if c in train.columns]]
    test = test[[c for c in expected_cols if c in test.columns]]
    
    train_X = ce.fit(train, train['click']).transform(train)
    test_X = ce.transform(test)
    
    train_y = train_X.pop('click')
    test_y = test_X.pop('click')
    
    # Drop columns not used for training
    cols_to_drop = ['hour', 'id']
    train_X = train_X.drop(columns=[c for c in cols_to_drop if c in train_X.columns])
    test_X = test_X.drop(columns=[c for c in cols_to_drop if c in test_X.columns])
    
    print("Handling imbalance...")
    weight = calculate_class_weights(train_y)
    print(f"Calculated scale_pos_weight: {weight:.2f}")
    
    print("Starting hyperparameter tuning...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cascade-ctr-prediction")
    
    study = optuna.create_study(direction="maximize")
    
    with mlflow.start_run(run_name="optuna_search"):
        study.optimize(lambda trial: objective(trial, config, train_X, train_y, test_X, test_y, weight), 
                       n_trials=config['tune']['n_trials'])
        
        print("Best trial:")
        print(study.best_trial.params)
        
        # Train best model
        best_params = study.best_trial.params
        best_params['objective'] = 'binary'
        best_params['scale_pos_weight'] = weight
        best_params['random_state'] = config['data']['random_state']
        best_params['verbose'] = -1
        
        best_model = get_lightgbm_model(best_params)
        best_model.fit(train_X, train_y)
        
        # Save model using MLflow
        mlflow.lightgbm.log_model(best_model, "best_lgbm_model")
        print("Best model trained and saved to MLflow.")

if __name__ == '__main__':
    run_pipeline()
