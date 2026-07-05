import pandas as pd
from category_encoders import TargetEncoder

class CategoricalEncoder:
    def __init__(self, config):
        self.config = config
        self.encoder = TargetEncoder()
        self.cat_cols = []
        
    def fit(self, df, y):
        # Identify categorical columns explicitly from config
        self.cat_cols = self.config['features']['user_cols'] + self.config['features']['ad_cols'] + self.config['features']['context_cols']
        if 'cross_cols' in self.config['features']:
            self.cat_cols += [f"{c[0]}_x_{c[1]}" for c in self.config['features']['cross_cols']]
        
        self.cat_cols = [c for c in self.cat_cols if c in df.columns and c != 'hour']
        
        if self.cat_cols:
            # Ensure they are treated as strings for target encoding
            for col in self.cat_cols:
                df[col] = df[col].astype(str)
            self.encoder.fit(df[self.cat_cols], y)
            
        return self
        
    def transform(self, df):
        df = df.copy()
        if self.cat_cols:
            for col in self.cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            encoded = self.encoder.transform(df[self.cat_cols])
            for col in self.cat_cols:
                df[col] = encoded[col]
        return df

if __name__ == '__main__':
    print("Encoder module ready.")
