#!/usr/bin/env bash
set -e

echo "Starting end-to-end pipeline..."

# 1. Scrape data
echo "Scraping data from eBay..."
python3 scripts/data_collection.py

# 2. Clean and prepare data
echo "Cleaning and preparing data..."
python3 scripts/data_cleaning.py

# 3. Train ResNet50
echo "Training ResNet50 model with DeepSpeed and Liger..."
deepspeed --num_gpus=1 scripts/train_resnet50.py --deepspeed_config scripts/utils/deepspeed_config.json

# 4. Fine-tune GPT-2 with LoRA
echo "Fine-tuning GPT-2 with LoRA and DeepSpeed..."
deepspeed --num_gpus=1 scripts/fine_tune_gpt2.py --deepspeed_config scripts/utils/deepspeed_config.json

# 5. Combine predictions (inference)
echo "Combining predictions..."
python3 scripts/combine_predictions.py --image data/images/sample.jpg

echo "Pipeline completed successfully!"
