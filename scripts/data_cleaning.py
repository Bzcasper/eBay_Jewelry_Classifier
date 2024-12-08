# scripts/data_cleaning.py

import os
import pandas as pd
from config import DATA_RAW_DIR, DATA_CLEANED_DIR
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(input_file='ebay_jewelry_data_cleaned.csv', output_file='ebay_jewelry_data_final.csv'):
    input_path = os.path.join(DATA_RAW_DIR, input_file)
    output_path = os.path.join(DATA_CLEANED_DIR, output_file)
    
    try:
        if not os.path.exists(DATA_CLEANED_DIR):
            os.makedirs(DATA_CLEANED_DIR)
        
        df = pd.read_csv(input_path)
        
        # Remove any remaining entries with missing values
        initial_count = df.shape[0]
        df.dropna(inplace=True)
        final_count = df.shape[0]
        logging.info(f"Dropped {initial_count - final_count} entries with missing values.")
        
        # Ensure price is float
        df['price'] = df['price'].astype(float)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        df.to_csv(output_path, index=False)
        logging.info(f"Final cleaned data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")

def main():
    clean_data()

if __name__ == '__main__':
    main()
