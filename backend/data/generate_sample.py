import pandas as pd
import numpy as np
import yaml
import os

def generate_sample_data(config_path='config.yaml', output_path='data/sample_dataset.csv'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    n_samples = config['data']['sample_size']
    np.random.seed(config['data']['random_state'])
    
    # Generate timestamp for ordering (last 30 days)
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=30)
    seconds_diff = (end_time - start_time).total_seconds()
    timestamps = start_time + pd.to_timedelta(np.random.uniform(0, seconds_diff, n_samples), unit='s')
    
    # Generate user data
    n_users = int(n_samples * 0.1)
    user_ids = [f'U{i:05d}' for i in range(n_users)]
    device_types = ['mobile', 'desktop', 'tablet']
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
    
    df_users = pd.DataFrame({
        'user_id': user_ids,
        'device_type': np.random.choice(device_types, n_users, p=[0.7, 0.2, 0.1]),
        'age_group': np.random.choice(age_groups, n_users)
    })
    
    # Generate ad data
    n_ads = int(n_samples * 0.05)
    ad_ids = [f'A{i:04d}' for i in range(n_ads)]
    campaign_ids = [f'C{i:03d}' for i in range(int(n_ads * 0.2))]
    categories = ['electronics', 'fashion', 'sports', 'home', 'automotive']
    
    df_ads = pd.DataFrame({
        'ad_id': ad_ids,
        'campaign_id': np.random.choice(campaign_ids, n_ads),
        'category': np.random.choice(categories, n_ads)
    })
    
    # Generate interactions
    data = {
        'timestamp': timestamps,
        'user_id': np.random.choice(user_ids, n_samples),
        'ad_id': np.random.choice(ad_ids, n_samples),
        'hour': timestamps.hour,
        'day_of_week': timestamps.dayofweek,
        'position': np.random.randint(1, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # ---------------------------------------------------------
    # Inject REAL patterns so the ML model can actually learn something!
    # ---------------------------------------------------------
    
    # Merge context with user and ad features
    df = df.merge(df_users, on='user_id', how='left')
    df = df.merge(df_ads, on='ad_id', how='left')
    
    # Simulate clicks with massive signals for a great visual demo
    base_prob = 0.05  # Base 5%
    
    # Signal 1: Mobile + Electronics = Huge CTR
    is_mobile_elec = (df['device_type'] == 'mobile') & (df['category'] == 'electronics')
    
    # Signal 2: Top position gets massive boost
    is_top_pos = df['position'] <= 2
    
    # Signal 3: Weekend + Fashion
    is_weekend_fashion = (df['day_of_week'] >= 5) & (df['category'] == 'fashion')
    
    probs = np.full(n_samples, base_prob)
    probs[is_mobile_elec] += 0.70  # Jumps to 75%
    probs[is_top_pos] += 0.15      # Jumps up 15%
    probs[is_weekend_fashion] += 0.40 # Jumps up 40%
    
    # Cap probabilities
    probs = np.clip(probs, 0.01, 0.95)
    
    # Generate target
    df['click'] = np.random.binomial(1, probs)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} rows with click rate: {df['click'].mean():.4f}")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    generate_sample_data()
