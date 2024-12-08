# scripts/data_collection.py

import requests
import pandas as pd
import os
from config import (
    EBAY_APP_ID,
    EBAY_FINDING_API_ENDPOINT,
    DATA_RAW_DIR
)
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_ebay_data(keywords, max_entries=100, page_number=1):
    params = {
        'OPERATION-NAME': 'findItemsByKeywords',
        'SERVICE-VERSION': '1.0.0',
        'SECURITY-APPNAME': EBAY_APP_ID,
        'RESPONSE-DATA-FORMAT': 'JSON',
        'keywords': keywords,
        'paginationInput.entriesPerPage': max_entries,
        'paginationInput.pageNumber': page_number,
        'outputSelector': 'PictureURLLarge'
    }
    
    try:
        response = requests.get(EBAY_FINDING_API_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        items = data.get('findItemsByKeywordsResponse', [{}])[0].get('searchResult', [{}])[0].get('item', [])
        
        records = []
        for item in items:
            title = item.get('title', [None])[0]
            price_info = item.get('sellingStatus', [{}])[0].get('currentPrice', [{}])
            price = price_info[0].get('__value__', None) if price_info else None
            image_url = item.get('galleryURL', [None])[0]
            if title and price and image_url:
                records.append({'title': title, 'price': float(price), 'image_url': image_url})
        
        return records
    except Exception as e:
        logging.error(f"Error fetching data from eBay API: {e}")
        return []

def save_data(df, filename='ebay_jewelry_data.csv'):
    try:
        if not os.path.exists(DATA_RAW_DIR):
            os.makedirs(DATA_RAW_DIR)
        filepath = os.path.join(DATA_RAW_DIR, filename)
        df.to_csv(filepath, index=False)
        logging.info(f"Data saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")

def collect_data(keywords='jewelry', total_pages=10):
    all_records = []
    for page in tqdm(range(1, total_pages + 1), desc="Fetching eBay data"):
        records = fetch_ebay_data(keywords, max_entries=100, page_number=page)
        all_records.extend(records)
        logging.info(f"Fetched page {page} with {len(records)} records.")
    
    if all_records:
        combined_df = pd.DataFrame(all_records)
        save_data(combined_df, 'ebay_jewelry_data.csv')
    else:
        logging.warning("No data fetched from eBay API.")

if __name__ == '__main__':
    collect_data()
