# config.py

import os

# eBay API Credentials
EBAY_APP_ID = 'YOUR_EBAY_APP_ID'  # Replace with your actual eBay App ID

# Directories
DATA_RAW_DIR = os.path.join('data', 'raw')
DATA_CLEANED_DIR = os.path.join('data', 'cleaned')
IMAGES_DIR = os.path.join('data', 'images')
DESCRIPTIONS_DIR = os.path.join('data', 'descriptions')
MODELS_DIR = 'models'
GPT2_MODEL_DIR = os.path.join(MODELS_DIR, 'gpt2-jewelry')
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, 'resnet50_jewelry.pth')

# eBay API Endpoint
EBAY_FINDING_API_ENDPOINT = 'https://svcs.ebay.com/services/search/FindingService/v1'

# Scraping Headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
                  '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Training Parameters
RESNET_EPOCHS = 10
GPT2_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
