import logging
from scripts.data_cleaning_class import JewelryDataCleaner
from config import RAW_DIR, CLEANED_DIR

# data_cleaning.py: Runs the cleaning, image downloading, dataset preparation steps.

if __name__ == '__main__':
    logging.info("Starting data cleaning process.")
    cleaner = JewelryDataCleaner(raw_dir=str(RAW_DIR), output_dir=str(CLEANED_DIR))
    df = cleaner.clean_listings()
    if not df.empty:
        df = cleaner.download_images(df)
        cleaner.prepare_training_datasets(df)
        logging.info("Data cleaning and preparation completed successfully.")
    else:
        logging.warning("No data to clean. Check scraping.")
