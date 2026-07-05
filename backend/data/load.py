import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import os

def load_data_and_split(config_path='config.yaml', data_path='data/raw/avazu_sample.csv'):
    """
    Loads data and performs a time-aware train/test split.
    Because this is CTR data, we must not predict the past using the future.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_full_path = os.path.join(base_dir, config_path)
    
    with open(config_full_path, 'r') as f:
        config = yaml.safe_load(f)
        
    data_full_path = os.path.join(base_dir, data_path)
    df = pd.read_csv(data_full_path)
    
    # Ensure it's sorted by time (Avazu uses 'hour' as YYMMDDHH)
    df = df.sort_values('hour').reset_index(drop=True)
    
    test_size = config['data']['test_size']
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df

if __name__ == '__main__':
    train, test = load_data_and_split()
    print(f"Train size: {len(train)}, Test size: {len(test)}")
