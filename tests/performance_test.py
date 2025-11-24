# performance_test.py
import pandas as pd
import sys
import time
import os
sys .path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     
from src.predict import predict_url

# Load URLs for performance testing
data_path = os.path.join("data", "urls_performance.csv")
urls = pd.read_csv(data_path)['url']

print(f"Testing performance on {len(urls)} URLs...\n")

start_time = time.time()

for url in urls:
    predict_url(url)

end_time = time.time()

total_time = end_time - start_time
avg_time = total_time / len(urls)

print(f"✅ Total URLs Processed: {len(urls)}")
print(f"⏱ Total Time: {total_time:.3f} seconds")
print(f"⚡ Average Time per URL: {avg_time:.5f} seconds")