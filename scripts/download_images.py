# scripts/download_images.py

import os
import requests
import pandas as pd
from config import DATA_RAW_DIR, IMAGES_DIR, HEADERS
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_image(url, filepath):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
    return False

def download_images(csv_file='ebay_jewelry_data.csv', output_csv='ebay_jewelry_data_cleaned.csv'):
    try:
        data_path = os.path.join(DATA_RAW_DIR, csv_file)
        df = pd.read_csv(data_path)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return
    
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    
    success_indices = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading images"):
        image_url = row['image_url']
        image_path = os.path.join(IMAGES_DIR, f"{index}.jpg")
        success = download_image(image_url, image_path)
        if success:
            success_indices.append(index)
    
    # Keep only the entries with successfully downloaded images
    cleaned_df = df.loc[success_indices].reset_index(drop=True)
    cleaned_data_path = os.path.join(DATA_RAW_DIR, output_csv)
    try:
        cleaned_df.to_csv(cleaned_data_path, index=False)
        logging.info(f"Cleaned data saved to {cleaned_data_path}")
    except Exception as e:
        logging.error(f"Error saving cleaned data: {e}")

def main():
    download_images()

if __name__ == '__main__':
    main()
