import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.user_stats = {}
        self.ad_stats = {}
        
    def fit(self, df):
        """
        Calculates historical click rates from the training data.
        """
        # User historical click rate
        self.user_stats = df.groupby('user_id')['click'].agg(['mean', 'count']).to_dict(orient='index')
        
        # Ad historical click rate
        self.ad_stats = df.groupby('ad_id')['click'].agg(['mean', 'count']).to_dict(orient='index')
        
        return self
        
    def transform(self, df):
        """
        Applies feature engineering to the data.
        """
        df = df.copy()
        
        # 1. Historical Click Rates (with smoothing for cold start)
        global_mean = 0.01 # Default fallback
        
        def get_smoothed_rate(id_val, stats_dict, global_mean, alpha=10):
            if id_val in stats_dict:
                stats = stats_dict[id_val]
                return (stats['mean'] * stats['count'] + global_mean * alpha) / (stats['count'] + alpha)
            return global_mean
            
        df['user_hist_ctr'] = df['user_id'].apply(lambda x: get_smoothed_rate(x, self.user_stats, global_mean))
        df['ad_hist_ctr'] = df['ad_id'].apply(lambda x: get_smoothed_rate(x, self.ad_stats, global_mean))
        
        # 2. Cross Features
        for cross in self.config['features']['cross_cols']:
            col_name = f"{cross[0]}_x_{cross[1]}"
            df[col_name] = df[cross[0]].astype(str) + "_" + df[cross[1]].astype(str)
            
        # 3. Time features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(int)
        
        return df

if __name__ == '__main__':
    print("Feature Engineer module ready.")
