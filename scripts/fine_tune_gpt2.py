# scripts/fine_tune_gpt2.py

import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from config import (
    DATA_CLEANED_DIR,
    GPT2_MODEL_DIR,
    GPT2_EPOCHS
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(file_path, tokenizer, block_size=128):
    try:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size
        )
    except Exception as e:
        logging.error(f"Error loading dataset from {file_path}: {e}")
        return None

def fine_tune_gpt2():
    try:
        # Initialize tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 does not have a pad token
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        logging.info("Loaded GPT-2 tokenizer and model.")
    except Exception as e:
        logging.error(f"Error loading GPT-2 model/tokenizer: {e}")
        return
    
    # Prepare datasets
    train_file = os.path.join(DATA_CLEANED_DIR, 'train', 'descriptions.txt')
    val_file = os.path.join(DATA_CLEANED_DIR, 'val', 'descriptions.txt')
    
    train_dataset = load_dataset(train_file, tokenizer)
    val_dataset = load_dataset(val_file, tokenizer)
    
    if not train_dataset or not val_dataset:
        logging.error("Failed to load datasets for GPT-2 fine-tuning.")
        return
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=GPT2_MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=GPT2_EPOCHS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=100,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Fine-tune the model
    try:
        trainer.train()
        logging.info("GPT-2 model fine-tuned successfully.")
    except Exception as e:
        logging.error(f"Error during GPT-2 fine-tuning: {e}")
        return
    
    # Save the fine-tuned model
    try:
        trainer.save_model(GPT2_MODEL_DIR)
        logging.info(f"GPT-2 model fine-tuned and saved to {GPT2_MODEL_DIR}")
    except Exception as e:
        logging.error(f"Error saving GPT-2 model: {e}")

def main():
    fine_tune_gpt2()

if __name__ == '__main__':
    main()
