import torch
import logging
import argparse
import json
import time
import numpy as np
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model
import deepspeed
from config import GPT2_CONFIG, CLEANED_DIR, MODELS_DIR
from scripts.utils.liger_optimizer import LigerOptimizer

def load_data():
    train_text_path=CLEANED_DIR/'llava'/'train.txt'
    if not train_text_path.exists():
        logging.warning(f"{train_text_path} not found, creating sample.")
        train_text_path.parent.mkdir(parents=True,exist_ok=True)
        train_text_path.write_text("Sample training text.\nAnother line of text.")
    train_text=train_text_path.read_text().strip().split('\n')
    logging.info(f"Loaded {len(train_text)} lines of data.")
    return train_text

def tokenize_data(tokenizer,texts,max_length=GPT2_CONFIG['max_length']):
    inputs=tokenizer('\n'.join(texts),return_tensors='pt',padding=True,truncation=True,max_length=max_length)
    return inputs['input_ids'], inputs.get('attention_mask',None)

def apply_lora_to_gpt2(model,rank):
    lora_config=LoraConfig(r=rank,lora_alpha=32,lora_dropout=0.1,bias='none',task_type='CAUSAL_LM')
    model=get_peft_model(model,lora_config)
    logging.info("LoRA layers added to GPT-2.")
    return model

def evaluate_perplexity(model_engine,input_ids,attention_mask,device,batch_size):
    model_engine.eval()
    losses=[]
    with torch.no_grad():
        for i in range(0,input_ids.size(0),batch_size):
            batch_ids=input_ids[i:i+batch_size].to(device)
            mask=attention_mask[i:i+batch_size].to(device) if attention_mask is not None else None
            outputs=model_engine(batch_ids,attention_mask=mask,labels=batch_ids)
            loss=outputs.loss.item()
            losses.append(loss)
    perplexity=float(np.exp(np.mean(losses)))
    return perplexity

def save_checkpoint(model,best_metric):
    checkpoint_dir=MODELS_DIR/'gpt2_lora_deepspeed_best'
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    model=model.merge_and_unload()
    model.save_pretrained(checkpoint_dir)
    tokenizer=GPT2Tokenizer.from_pretrained(GPT2_CONFIG['model_name'])
    tokenizer.save_pretrained(checkpoint_dir)
    metrics_path=checkpoint_dir/'metrics.json'
    with metrics_path.open('w')as f:
        json.dump({'best_val_perplexity':best_metric},f,indent=2)
    logging.info(f"Best model saved with Val Perplexity {best_metric:.2f}")

def train_model(model_engine,input_ids,attention_mask,device,batch_size,epochs):
    best_metric=float('inf')
    epochs_no_improve=0
    early_stopping_patience=1
    logging.info(f"GPT-2 training {epochs} epochs CPU LoRA Liger DeepSpeed.")
    start=time.time()
    for epoch in range(epochs):
        logging.debug(f"Epoch {epoch+1}/{epochs}")
        model_engine.train()
        losses=[]
        for i in range(0,input_ids.size(0),batch_size):
            batch_ids=input_ids[i:i+batch_size].to(device)
            mask=attention_mask[i:i+batch_size].to(device) if attention_mask is not None else None
            model_engine.optimizer.zero_grad()
            outputs=model_engine(batch_ids,attention_mask=mask,labels=batch_ids)
            loss=outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            losses.append(loss.item())
        train_loss=np.mean(losses)
        val_perp=evaluate_perplexity(model_engine,input_ids,attention_mask,device,batch_size)
        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss:{train_loss:.4f}, Val PPL:{val_perp:.2f}")
        if val_perp<best_metric:
            best_metric=val_perp
            epochs_no_improve=0
            model=model_engine.module
            save_checkpoint(model,best_metric)
        else:
            epochs_no_improve+=1
            if epochs_no_improve>=early_stopping_patience:
                logging.info("No improvement in perplexity, stopping.")
                break
    time_elapsed=time.time()-start
    logging.info(f"Training complete in {time_elapsed:.2f}s. Best PPL:{best_metric:.2f}")

def run_training(batch_size,epochs,device,deepspeed_config):
    tokenizer=GPT2Tokenizer.from_pretrained(GPT2_CONFIG['model_name'])
    tokenizer.pad_token=tokenizer.eos_token
    model=GPT2LMHeadModel.from_pretrained(GPT2_CONFIG['model_name'])
    train_data=load_data()
    if not train_data:
        logging.error("No training data.")
        return
    input_ids,attention_mask=tokenize_data(tokenizer,train_data)
    model=apply_lora_to_gpt2(model,GPT2_CONFIG['lora_rank'])
    optimizer=LigerOptimizer(model.parameters(),lr=GPT2_CONFIG['learning_rate'],weight_decay=0.01)
    model_engine,optimizer,_,_=deepspeed.initialize(model=model,optimizer=optimizer,model_parameters=model.parameters(),config=deepspeed_config)
    train_model(model_engine,input_ids,attention_mask,device,batch_size,epochs)

if __name__=='__main__':
    logging.info("Starting GPT-2 fine-tuning.")
    parser=argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config",type=str,default="scripts/utils/deepspeed_config.json")
    args=parser.parse_args()
    device=torch.device('cpu')

    # Primary config
    primary_batch_size=GPT2_CONFIG['batch_size']
    primary_epochs=GPT2_CONFIG['epochs']
    fallback_batch_size=max(1,primary_batch_size//2)
    fallback_epochs=max(1,primary_epochs//2)
    try:
        run_training(primary_batch_size,primary_epochs,device,args.deepspeed_config)
        logging.info("GPT-2 training (primary) completed.")
    except Exception as e:
        logging.error(f"Primary training failed: {e}")
        logging.info(f"Trying fallback with batch_size={fallback_batch_size}, epochs={fallback_epochs}")
        try:
            run_training(fallback_batch_size,fallback_epochs,device,args.deepspeed_config)
            logging.info("GPT-2 training (fallback) completed.")
        except Exception as e2:
            logging.critical(f"Fallback also failed: {e2}")
            logging.info("No more fallbacks, aborting.")
