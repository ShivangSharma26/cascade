import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.user_stats = {}
        self.ad_stats = {}
        
    def fit(self, df):
        """
        No need to pre-compute historical CTRs to prevent data leakage.
        LightGBM will learn the patterns robustly.
        """
        return self
        
    def transform(self, df):
        """
        Applies feature engineering to the data.
        """
        df = df.copy()
        
        # 1. Parse YYMMDDHH into datetime
        if 'hour' in df.columns and df['hour'].max() > 1000000:
            dt = pd.to_datetime(df['hour'], format='%y%m%d%H')
            df['hour_of_day'] = dt.dt.hour
            df['day_of_week'] = dt.dt.dayofweek
        elif 'hour' in df.columns:
            df['hour_of_day'] = df['hour'].astype(int)
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = 3
        else:
            # Fallback if already parsed
            df['hour_of_day'] = 12
            df['day_of_week'] = 3
            
        # 2. Cross Features
        for cross in self.config['features'].get('cross_cols', []):
            col_name = f"{cross[0]}_x_{cross[1]}"
            if cross[0] in df.columns and cross[1] in df.columns:
                df[col_name] = df[cross[0]].astype(str) + "_" + df[cross[1]].astype(str)
            
        # 3. Time features
        if 'day_of_week' in df.columns:
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        if 'hour_of_day' in df.columns:
            df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 4)).astype(int)
            
        return df

if __name__ == '__main__':
    print("Feature Engineer module ready.")
