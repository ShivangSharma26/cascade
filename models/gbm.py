import lightgbm as lgb
import xgboost as xgb

def get_lightgbm_model(config_params):
    """
    Returns a LightGBM Classifier.
    """
    model = lgb.LGBMClassifier(**config_params)
    return model

def get_xgboost_model(config_params):
    """
    Returns an XGBoost Classifier.
    """
    model = xgb.XGBClassifier(**config_params)
    return model

if __name__ == '__main__':
    print("GBM models defined.")
