import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import colorlog

# Load environment variables for eBay API keys or other creds
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RAW_DIR = DATA_DIR / 'raw'
CLEANED_DIR = DATA_DIR / 'cleaned'
PROCESSED_DIR = DATA_DIR / 'processed'
IMAGES_DIR = DATA_DIR / 'images'

for d in [RAW_DIR, CLEANED_DIR, PROCESSED_DIR, IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# eBay API Credentials or any other keys
EBAY_APP_ID = os.getenv("EBAY_APP_ID", "")
EBAY_DEV_ID = os.getenv("EBAY_DEV_ID", "")
EBAY_CERT_ID = os.getenv("EBAY_CERT_ID", "")
EBAY_TOKEN = os.getenv("EBAY_TOKEN", "")

SCRAPING_CONFIG = {
    'max_pages_per_category': 2,
    'delay_between_requests': 2.0,
    'timeout': 10,
    'max_retries': 3
}

JEWELRY_CATEGORIES = [
    'necklace', 'pendant', 'bracelet', 'wristwatch', 'earring', 'ring'
]

SUBCATEGORIES = {
    'necklace': ['gold necklace', 'silver necklace'],
    'pendant': ['diamond pendant', 'ruby pendant'],
    'bracelet': ['gold bracelet', 'tennis bracelet'],
    'wristwatch': ['automatic watch', 'quartz watch'],
    'earring': ['stud earrings', 'hoop earrings'],
    'ring': ['engagement ring', 'fashion ring']
}

# CPU-friendly training configs
RESNET_CONFIG = {
    'batch_size': 4,
    'epochs': 1,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'momentum': 0.9,
    'num_workers': 0,
    'pin_memory': False,
    'early_stopping_patience': 1
}

GPT2_CONFIG = {
    'model_name': 'gpt2',
    'batch_size': 1,
    'epochs': 1,
    'learning_rate': 2e-5,
    'warmup_steps': 20,
    'max_length': 128,
    'temperature': 0.7,
    'top_p': 0.9,
    'lora_rank': 4
}

# Logging configuration with colorlog
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = (
    "%(asctime)s "
    "%(log_color)s%(levelname)-8s%(reset)s "
    "%(cyan)s[%(name)s]%(reset)s "
    "%(message_log_color)s%(message)s%(reset)s"
)
LOG_COLORS = {
    'DEBUG': 'white',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white'
}
MESSAGE_COLORS = {
    'DEBUG': 'white',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red'
}
formatter = colorlog.ColoredFormatter(
    LOG_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors=LOG_COLORS,
    secondary_log_colors={'message': MESSAGE_COLORS}
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logging.root.setLevel(LOG_LEVEL)
logging.root.handlers = [handler]
