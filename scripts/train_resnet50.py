import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models, transforms
from pathlib import Path
import logging
import numpy as np
import time
import json
from PIL import Image
import argparse
import deepspeed
from config import CLEANED_DIR, MODELS_DIR, RESNET_CONFIG
from scripts.utils.liger_optimizer import LigerOptimizer
from peft import LoraConfig, get_peft_model

def pil_loader(path: str) -> Image.Image:
    with open(path,'rb') as f:
        img=Image.open(f).convert('RGB')
    return img

def load_data():
    train_dir=CLEANED_DIR/'resnet'/'train'
    val_dir=CLEANED_DIR/'resnet'/'val'
    if not train_dir.exists() or not val_dir.exists():
        logging.error("Training/Validation dirs not found. Run data_cleaning first.")
        return None,None,None
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_dataset=data.DatasetFolder(str(train_dir),loader=pil_loader,extensions=("jpg",),transform=transform)
    val_dataset=data.DatasetFolder(str(val_dir),loader=pil_loader,extensions=("jpg",),transform=transform)
    train_loader=data.DataLoader(train_dataset,batch_size=RESNET_CONFIG['batch_size'],shuffle=True,num_workers=RESNET_CONFIG['num_workers'],pin_memory=RESNET_CONFIG['pin_memory'])
    val_loader=data.DataLoader(val_dataset,batch_size=RESNET_CONFIG['batch_size'],shuffle=False,num_workers=RESNET_CONFIG['num_workers'],pin_memory=RESNET_CONFIG['pin_memory'])
    return train_loader,val_loader,len(train_dataset.classes)

def apply_lora_to_resnet(model: nn.Module, rank: int):
    # LoRA on final fc layer as demonstration.
    lora_config=LoraConfig(r=rank,lora_alpha=32,lora_dropout=0.1,bias='none',task_type='CAUSAL_LM')
    model=get_peft_model(model,lora_config)
    logging.info("LoRA layers added to ResNet-50 final layer.")
    return model

def validate_model(model, val_loader, engine, device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for inputs,labels in val_loader:
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=engine(inputs)
            _,preds=torch.max(outputs,1)
            correct+=torch.sum(preds==labels).item()
            total+=labels.size(0)
    return correct/total if total>0 else 0.0

def save_checkpoint(model,best_acc):
    checkpoint_dir=MODELS_DIR/'resnet50_lora_deepspeed_best'
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_path=checkpoint_dir/'resnet50_best.pth'
    model=model.merge_and_unload()
    torch.save(model.state_dict(),checkpoint_path)
    logging.info(f"Best model saved to {checkpoint_path} with Val Acc: {best_acc:.4f}")
    metrics_path=checkpoint_dir/'metrics.json'
    with metrics_path.open('w')as f:
        json.dump({'best_val_acc':best_acc},f,indent=2)
    logging.debug(f"Metrics saved to {metrics_path}")

def train_model(train_loader,val_loader,num_classes,engine,device):
    criterion=nn.CrossEntropyLoss().to(device)
    best_acc=0.0
    epochs_no_improve=0
    early_stopping=RESNET_CONFIG['early_stopping_patience']
    epochs=RESNET_CONFIG['epochs']
    logging.info(f"Training ResNet50 for {epochs} epochs on CPU with LoRA, Liger, DeepSpeed.")
    since=time.time()
    model=engine.module
    for epoch in range(epochs):
        logging.debug(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss=0.0
        for inputs,labels in train_loader:
            inputs,labels=inputs.to(device),labels.to(device)
            engine.optimizer.zero_grad()
            outputs=engine(inputs)
            loss=criterion(outputs,labels)
            engine.backward(loss)
            engine.step()
            running_loss+=loss.item()*inputs.size(0)
        train_loss=running_loss/len(train_loader.dataset)
        val_acc=validate_model(model,val_loader,engine,device)
        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc>best_acc:
            best_acc=val_acc
            epochs_no_improve=0
            save_checkpoint(model,best_acc)
        else:
            epochs_no_improve+=1
            if epochs_no_improve>=early_stopping:
                logging.info("No improvement, early stopping.")
                break
    time_elapsed=time.time()-since
    logging.info(f"Training complete in {time_elapsed:.2f}s. Best Val Acc:{best_acc:.4f}")

if __name__=='__main__':
    logging.info("Loading data for ResNet50 training...")
    train_loader,val_loader,num_classes=load_data()
    if train_loader is None or val_loader is None or num_classes is None:
        logging.error("Data loading failed.")
        exit(1)

    parser=argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config",type=str,default="scripts/utils/deepspeed_config.json")
    args=parser.parse_args()

    device=torch.device('cpu')
    model=models.resnet50(pretrained=False)
    num_ftrs=model.fc.in_features
    model.fc=nn.Linear(num_ftrs,num_classes)
    from deepspeed import initialize
    model=apply_lora_to_resnet(model,rank=4)
    optimizer=LigerOptimizer(model.parameters(),lr=RESNET_CONFIG['learning_rate'],weight_decay=RESNET_CONFIG['weight_decay'])
    model_engine,optimizer,_,_=initialize(model=model,optimizer=optimizer,model_parameters=model.parameters(),config=args.deepspeed_config)
    train_model(train_loader,val_loader,num_classes,model_engine,device)
    logging.info("ResNet50 training script completed.")
