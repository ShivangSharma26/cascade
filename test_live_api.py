import requests
import time
import sys

def test_api():
    url = "http://localhost:8001/predict"
    payload = {
        "user_id": "U00001",
        "ad_id": "A0001",
        "campaign_id": "C001",
        "category": "fashion",
        "device_type": "mobile",
        "age_group": "25-34",
        "hour": 20,
        "day_of_week": 5,
        "position": 1
    }
    
    # Try connecting a few times in case the server is still booting
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=20)
            if response.status_code == 200:
                print("SUCCESS: Connected to API!")
                print("Response:", response.json())
                return
            else:
                print(f"Server responded with {response.status_code}")
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            print(f"Waiting for server to start... (Attempt {i+1}/{max_retries})")
            time.sleep(2)
            
    print("FAILED: Could not connect to API server.")
    sys.exit(1)

if __name__ == "__main__":
    test_api()
