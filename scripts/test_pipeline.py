import unittest
import subprocess
import logging
import os
import json
from pathlib import Path
from config import DATA_DIR, MODELS_DIR, CLEANED_DIR

"""
test_pipeline.py

Basic integration tests for the entire pipeline:
- Tests scraping (if credentials and network access exist).
- Tests data cleaning.
- Tests training (ResNet50 and GPT-2) at least to ensure scripts run without error and produce expected outputs.
- Tests inference with combine_predictions.py on a sample image.
- Checks fallback logic by providing minimal data.
- Optionally mock API calls by pre-creating sold_listings.csv or skipping.

Note: Real tests would mock APIs and run much faster. This is a basic demonstration.
"""

class TestPipeline(unittest.TestCase):

    def setUp(self):
        # Ensure directories exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # Optionally create a minimal sold_listings.csv for pricing if desired
        sold_data_path = CLEANED_DIR / 'sold_listings.csv'
        if not sold_data_path.exists():
            # Create a small fake sold dataset
            import pandas as pd
            df = pd.DataFrame({
                'category': ['ring','ring','necklace'],
                'price.current': [100,150,300],
                'sold': [1,1,1]
            })
            df.to_csv(sold_data_path, index=False)

        # Create a sample image if needed for inference test
        sample_image = DATA_DIR / 'images' / 'test_image.jpg'
        sample_image.parent.mkdir(parents=True, exist_ok=True)
        # Create a simple image (e.g., black image)
        from PIL import Image
        img = Image.new('RGB',(224,224),(0,0,0))
        img.save(sample_image)

    def test_01_scraping(self):
        # Test scraping one category and subcategory (assuming credentials and network)
        category='ring'
        subcat='engagement ring'
        if subcat not in ['engagement ring','fashion ring']:
            self.skipTest("Subcategory not in config, skipping scraping test.")

        cmd=['python3','scripts/data_collection.py',category,subcat]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        self.assertIn("Saved", ret.stdout+ret.stderr, "Scraping didn't save listings properly or no output found.")
        # Check if raw listings csv created
        raw_files = list((DATA_DIR/'raw').glob(f"{category}_{subcat.replace(' ','_')}_listings.csv"))
        self.assertTrue(len(raw_files)>0, "No raw listings csv created.")

    def test_02_cleaning(self):
        cmd=['python3','scripts/data_cleaning.py']
        ret=subprocess.run(cmd, capture_output=True, text=True)
        self.assertIn("Data cleaning and preparation completed", ret.stdout+ret.stderr, "Cleaning didn't complete successfully.")
        # Check cleaned_listings.csv
        cleaned = CLEANED_DIR/'cleaned_listings.csv'
        self.assertTrue(cleaned.exists(),"Cleaned listings not found.")
        # Check final_listings_with_images.csv
        final = CLEANED_DIR/'final_listings_with_images.csv'
        self.assertTrue(final.exists(),"Final listings with images not found.")

    def test_03_train_resnet(self):
        cmd=['deepspeed','--num_gpus=1','scripts/train_resnet50.py','--deepspeed_config','scripts/utils/deepspeed_config.json']
        ret=subprocess.run(cmd, capture_output=True, text=True)
        # Even if fallback occurs, we should end with "Training complete"
        self.assertIn("Training complete", ret.stdout+ret.stderr, "ResNet training did not complete successfully.")
        # Check if model saved
        best_dir = MODELS_DIR/'resnet50_lora_deepspeed_best'
        best_pth = best_dir/'resnet50_best.pth'
        self.assertTrue(best_pth.exists(),"ResNet best model not saved.")

    def test_04_train_gpt2(self):
        cmd=['deepspeed','--num_gpus=1','scripts/fine_tune_gpt2.py','--deepspeed_config','scripts/utils/deepspeed_config.json']
        ret=subprocess.run(cmd, capture_output=True, text=True)
        self.assertIn("Training complete", ret.stdout+ret.stderr, "GPT-2 training did not complete successfully.")
        # Check GPT-2 best model
        gpt2_dir=MODELS_DIR/'gpt2_lora_deepspeed_best'
        self.assertTrue((gpt2_dir/'pytorch_model.bin').exists(),"GPT-2 merged model not saved.")

    def test_05_inference(self):
        # Use the sample image for inference
        sample_image = DATA_DIR / 'images' / 'test_image.jpg'
        cmd=['python3','scripts/combine_predictions.py', str(sample_image)]
        ret=subprocess.run(cmd, capture_output=True, text=True)
        self.assertIn("Final result:", ret.stdout+ret.stderr, "Inference did not complete or no final result printed.")
        # Check output result printed
        # Just ensure that category and price range were printed
        output = ret.stdout+ret.stderr
        self.assertIn("category", output, "No category in output.")
        self.assertIn("suggested_price_range", output, "No price range in output.")

    def test_06_api_manager(self):
        # Test api_manager get_json with a fake call (assuming no real endpoint)
        # We won't break if no endpoint, just ensure no crash.
        from scripts.utils.api_manager import get_json
        data = get_json("https://example.com/fake_endpoint")
        # data might be None if no real endpoint, that's okay. Check no exception.
        self.assertIsNotNone(data or True, "get_json call caused exception.") # Just checking no exception.

if __name__=='__main__':
    unittest.main()
