import pandas as pd
from category_encoders import TargetEncoder

class CategoricalEncoder:
    def __init__(self, config):
        self.config = config
        self.encoder = TargetEncoder()
        self.cat_cols = []
        
    def fit(self, df, y):
        # Identify categorical columns
        self.cat_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
        
        if self.cat_cols:
            self.encoder.fit(df[self.cat_cols], y)
            
        return self
        
    def transform(self, df):
        df = df.copy()
        if self.cat_cols:
            encoded = self.encoder.transform(df[self.cat_cols])
            for col in self.cat_cols:
                df[col] = encoded[col]
        return df

if __name__ == '__main__':
    print("Encoder module ready.")
