import numpy as np

def calculate_class_weights(y):
    """
    Calculates scale_pos_weight for LightGBM/XGBoost.
    It's the ratio of negative class to positive class.
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    
    if n_pos == 0:
        return 1.0
        
    return n_neg / n_pos

def update_config_with_weights(config, y_train):
    """
    Updates the model config with the calculated scale_pos_weight.
    """
    weight = calculate_class_weights(y_train)
    config['model']['lightgbm_params']['scale_pos_weight'] = weight
    config['model']['xgboost_params']['scale_pos_weight'] = weight
    
    print(f"Set scale_pos_weight to {weight:.2f}")
    return config

if __name__ == '__main__':
    print("Imbalance handler ready.")
