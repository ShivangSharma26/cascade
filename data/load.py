import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_data_and_split(config_path='config.yaml', data_path='data/sample_dataset.csv'):
    """
    Loads data and performs a time-aware train/test split.
    Because this is CTR data, we must not predict the past using the future.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    df = pd.read_csv(data_path)
    
    # Ensure it's sorted by time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    test_size = config['data']['test_size']
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df

if __name__ == '__main__':
    train, test = load_data_and_split()
    print(f"Train size: {len(train)}, Test size: {len(test)}")
