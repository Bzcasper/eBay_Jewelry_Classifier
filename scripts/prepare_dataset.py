# scripts/prepare_dataset.py

import os
import pandas as pd
from config import DATA_CLEANED_DIR, IMAGES_DIR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def infer_category(title):
    title = title.lower()
    if 'ring' in title:
        return 'rings'
    elif 'necklace' in title:
        return 'necklaces'
    elif 'earring' in title:
        return 'earrings'
    elif 'bracelet' in title:
        return 'bracelets'
    elif 'pendant' in title:
        return 'pendants'
    else:
        return 'others'

def organize_dataset():
    cleaned_data_path = os.path.join(DATA_CLEANED_DIR, 'ebay_jewelry_data_final.csv')
    try:
        df = pd.read_csv(cleaned_data_path)
    except Exception as e:
        logging.error(f"Error reading cleaned CSV file: {e}")
        return
    
    # Infer categories
    df['category'] = df['title'].apply(infer_category)
    logging.info("Inferred categories based on titles.")
    
    # Split into train and validation
    try:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
        logging.info(f"Split data into {train_df.shape[0]} training and {val_df.shape[0]} validation samples.")
    except Exception as e:
        logging.error(f"Error during train-test split: {e}")
        return
    
    # Function to save images into class directories
    def save_images(df_subset, subset='train'):
        for index, row in tqdm(df_subset.iterrows(), total=df_subset.shape[0], desc=f"Saving {subset} images"):
            category = row['category']
            image_filename = f"{index}.jpg"
            src_image_path = os.path.join(IMAGES_DIR, image_filename)
            class_dir = os.path.join(DATA_CLEANED_DIR, subset, category)
            os.makedirs(class_dir, exist_ok=True)
            if os.path.exists(src_image_path):
                dest_path = os.path.join(class_dir, image_filename)
                try:
                    os.rename(src_image_path, dest_path)
                except Exception as e:
                    logging.error(f"Error moving {src_image_path} to {dest_path}: {e}")
    
    save_images(train_df, 'train')
    save_images(val_df, 'val')
    logging.info("Saved images into train and val directories.")
    
    # Save descriptions for GPT-2 training
    def save_descriptions(df_subset, subset='train'):
        descriptions_path = os.path.join(DATA_CLEANED_DIR, subset, 'descriptions.txt')
        try:
            with open(descriptions_path, 'w', encoding='utf-8') as f:
                for _, row in df_subset.iterrows():
                    description = f"Title: {row['title']}\nPrice: \nDescription:"
                    f.write(description + "\n")
            logging.info(f"Saved descriptions to {descriptions_path}")
        except Exception as e:
            logging.error(f"Error writing descriptions to {descriptions_path}: {e}")
    
    save_descriptions(train_df, 'train')
    save_descriptions(val_df, 'val')
    logging.info("Saved descriptions for GPT-2 fine-tuning.")
    
    logging.info("Dataset preparation completed.")

def main():
    organize_dataset()

if __name__ == '__main__':
    main()
