import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Any, List, Dict, Tuple
from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor
import io
import hashlib
import shutil
import random
import time
from config import CLEANED_DIR, IMAGES_DIR

class JewelryDataCleaner:
    def __init__(self, raw_dir: str = 'data/raw', output_dir: str = 'data/cleaned'):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'total_listings': 0,
            'cleaned_listings': 0,
            'downloaded_images': 0,
            'errors': []
        }

    def clean_listings(self) -> pd.DataFrame:
        csv_files = list(self.raw_dir.glob('*_listings.csv'))
        if not csv_files:
            logging.warning("No raw listings found.")
            return pd.DataFrame()

        logging.info(f"Combining {len(csv_files)} CSV files...")
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
        combined_df = pd.concat(dfs, ignore_index=True)
        self.stats['total_listings'] = len(combined_df)
        cleaned_df = self._clean_dataframe(combined_df)
        self.stats['cleaned_listings'] = len(cleaned_df)
        cleaned_df.to_csv(self.output_dir / 'cleaned_listings.csv', index=False)
        logging.info(f"Saved cleaned listings ({len(cleaned_df)}) to cleaned_listings.csv")
        return cleaned_df

    def _clean_price(self, x: Any) -> float:
        if pd.isnull(x):
            return np.nan
        try:
            return float(str(x).replace('$','').replace(',','').strip())
        except:
            return np.nan

    def _extract_materials(self, title: str) -> str:
        if pd.isnull(title):
            return ''
        title_lower = title.lower()
        materials = ['gold','silver','diamond','platinum','pearl','ruby','sapphire','emerald']
        found = [m for m in materials if m in title_lower]
        return ','.join(found)

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=['url'], keep='first')
        df['price.current'] = df['price'].apply(self._clean_price)
        df['category'] = df['category'].str.lower().fillna('unknown')
        df['condition'] = df['condition'].astype(str).str.lower().str.strip()
        df['materials'] = df['title'].apply(self._extract_materials)
        if 'scraped_at' in df.columns:
            df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
        df = df.dropna(subset=['title','price.current','category'])
        logging.info(f"Cleaned down to {len(df)} listings.")
        return df

    def _parse_image_urls(self, images_col: Any) -> List[str]:
        if pd.isnull(images_col):
            return []
        if isinstance(images_col, str):
            try:
                urls = json.loads(images_col)
                if isinstance(urls,list):
                    return urls
            except:
                return [x.strip() for x in images_col.split(',') if x.strip()]
        elif isinstance(images_col,list):
            return images_col
        return []

    def _download_and_save_image(self, url: str, category: str, idx: int) -> Any:
        if not url:
            return None
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            hash_str = hashlib.md5(url.encode()).hexdigest()[:10]
            filename = f"{category}_{idx}_{hash_str}.jpg"
            save_path = self.images_dir / filename
            img.convert('RGB').save(save_path,'JPEG',quality=95)
            return filename
        except Exception as e:
            err_msg = f"Error downloading {url}: {e}"
            self.stats['errors'].append(err_msg)
            logging.error(err_msg)
            return None

    def download_images(self, df: pd.DataFrame, max_workers: int=4) -> pd.DataFrame:
        df = df.copy()
        df['local_image_path'] = None
        logging.info(f"Downloading images for {len(df)} listings...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures={}
            for idx,row in df.iterrows():
                image_urls = self._parse_image_urls(row.get('images',''))
                if not image_urls:
                    continue
                futures[executor.submit(self._download_and_save_image, image_urls[0], row['category'], idx)] = idx
            for future in futures:
                res = future.result()
                idx = futures[future]
                if res is not None:
                    df.at[idx,'local_image_path']=res
                    self.stats['downloaded_images']+=1
        df = df.dropna(subset=['local_image_path'])
        df.to_csv(self.output_dir / 'final_listings_with_images.csv', index=False)
        logging.info(f"Downloaded {self.stats['downloaded_images']} images.")
        return df

    def prepare_training_datasets(self, df: pd.DataFrame):
        llava_dir = self.output_dir / 'llava'
        resnet_dir = self.output_dir / 'resnet'
        llava_dir.mkdir(exist_ok=True)
        resnet_dir.mkdir(exist_ok=True)
        train_df,val_df,test_df = self._split_data(df)
        self._prepare_llava_dataset(train_df,val_df,test_df,llava_dir)
        self._prepare_resnet_dataset(train_df,val_df,test_df,resnet_dir)
        with (self.output_dir / 'cleaning_stats.json').open('w') as f:
            json.dump(self.stats,f,indent=2)
        logging.info("Training datasets prepared.")

    def _split_data(self, df: pd.DataFrame, split_ratio=(0.7,0.15,0.15)) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        train_ratio,val_ratio,test_ratio=split_ratio
        train_df,val_df,test_df=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        for cat in df['category'].unique():
            cat_df=df[df['category']==cat].sample(frac=1,random_state=42)
            n=len(cat_df)
            n_train=int(n*train_ratio)
            n_val=int(n*val_ratio)
            cat_train=cat_df.iloc[:n_train]
            cat_val=cat_df.iloc[n_train:n_train+n_val]
            cat_test=cat_df.iloc[n_train+n_val:]
            train_df=pd.concat([train_df,cat_train])
            val_df=pd.concat([val_df,cat_val])
            test_df=pd.concat([test_df,cat_test])
        logging.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df,val_df,test_df

    def _prepare_llava_dataset(self, train_df, val_df, test_df, llava_dir:Path):
        for split in ['train','val','test']:
            (llava_dir / split / 'images').mkdir(parents=True,exist_ok=True)
        annotations=[]
        for split_df,split_name in [(train_df,'train'),(val_df,'val'),(test_df,'test')]:
            for idx,row in split_df.iterrows():
                src_path=self.images_dir/row['local_image_path']
                if not src_path.exists():
                    continue
                dst_path=llava_dir/split_name/'images'/f"{idx}.jpg"
                shutil.copy2(src_path,dst_path)
                annotation={
                    'image_id':idx,
                    'image_path':str(dst_path.relative_to(llava_dir)),
                    'title':row['title'],
                    'price':row['price.current'],
                    'category':row['category'],
                    'condition':row['condition'],
                    'split':split_name,
                    'conversations':self._create_llava_conversation(row)
                }
                annotations.append(annotation)
        with (llava_dir/'annotations.json').open('w')as f:
            json.dump(annotations,f,indent=2)
        logging.info(f"LLaVA dataset with {len(annotations)} annotations.")

    def _prepare_resnet_dataset(self, train_df, val_df, test_df, resnet_dir:Path):
        categories = pd.concat([train_df['category'], val_df['category'], test_df['category']]).unique()
        for split in ['train','val','test']:
            for cat in categories:
                (resnet_dir/split/cat).mkdir(parents=True,exist_ok=True)
        for split_df, split_name in [(train_df,'train'),(val_df,'val'),(test_df,'test')]:
            count=0
            for idx,row in split_df.iterrows():
                src_path=self.images_dir/row['local_image_path']
                if src_path.exists():
                    dst_path=resnet_dir/split_name/row['category']/f"{idx}.jpg"
                    shutil.copy2(src_path,dst_path)
                    count+=1
            logging.info(f"Prepared {count} images for ResNet {split_name} set.")

    def _create_llava_conversation(self, row) -> List[Dict[str,str]]:
        return [
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":f"Describe this {row['category']} jewelry item."},
            {"role":"assistant","content":f"This {row['category']} is '{row['title']}' priced at {row['price.current']}. Condition: {row['condition']}."}
        ]
