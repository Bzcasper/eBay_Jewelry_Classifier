import logging
import json
import time
from pathlib import Path
from typing import Dict, Any
import albumentations as A
import cv2
import numpy as np
from PIL import Image
from config import CLEANED_DIR, PROCESSED_DIR

# data_augmentation.py: Augments images for the dataset.
# CPU-only, uses albumentations, logs actions, fallback logic if needed (not much fallback needed here).

class JewelryAugmentor:
    def __init__(self, 
                 input_dir: str = str((CLEANED_DIR/'images')), 
                 output_dir: str = str((PROCESSED_DIR/'augmented')),
                 samples_per_image: int = 2):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.samples_per_image = samples_per_image
        self.output_dir.mkdir(parents=True,exist_ok=True)

        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1,p=0.5),
            A.GaussNoise(var_limit=(10.0,50.0),p=0.3),
            A.GaussianBlur(blur_limit=3,p=0.3)
        ])

        self.stats = {
            'total_processed':0,
            'total_augmented':0,
            'errors': []
        }

    def augment_dataset(self):
        image_paths = list(self.input_dir.glob('**/*.jpg'))
        if not image_paths:
            logging.warning(f"No .jpg images found in {self.input_dir}")
            return
        logging.info(f"Augmenting {len(image_paths)} images, {self.samples_per_image} samples each.")
        for img_path in image_paths:
            try:
                self._augment_single_image(img_path)
            except Exception as e:
                err_msg=f"Error augmenting {img_path.name}: {e}"
                self.stats['errors'].append(err_msg)
                logging.error(err_msg)

        stats_path=self.output_dir/'augmentation_stats.json'
        with stats_path.open('w') as f:
            json.dump(self.stats,f,indent=2)
        logging.info(f"Augmentation completed. Stats in {stats_path}")

    def _augment_single_image(self, img_path: Path):
        img=cv2.imread(str(img_path))
        if img is None:
            err_msg=f"Failed to load {img_path}"
            self.stats['errors'].append(err_msg)
            logging.error(err_msg)
            return
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        base_name=img_path.stem

        for i in range(self.samples_per_image):
            augmented=self.transform(image=img)['image']
            out_name=f"{base_name}_aug{i}.jpg"
            out_path=self.output_dir/out_name
            Image.fromarray(augmented).save(out_path,quality=95)
            self.stats['total_augmented']+=1
        self.stats['total_processed']+=1
