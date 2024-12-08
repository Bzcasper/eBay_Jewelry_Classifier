# scripts/train_resnet50.py

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from config import (
    DATA_CLEANED_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    RESNET_EPOCHS,
    RESNET_MODEL_PATH,
    MODELS_DIR
)
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_resnet50():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    train_dir = os.path.join(DATA_CLEANED_DIR, 'train')
    val_dir = os.path.join(DATA_CLEANED_DIR, 'val')
    
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
        logging.info("Loaded training and validation datasets.")
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load pre-trained ResNet-50
    model = models.resnet50(pretrained=True)
    
    # Modify the final layer
    num_ftrs = model.fc.in_features
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(num_ftrs, num_classes)
    logging.info(f"Modified ResNet-50 final layer to {num_classes} classes.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logging.info(f"Using device: {device}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logging.info("Initialized loss function and optimizer.")
    
    # Training loop
    for epoch in range(RESNET_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{RESNET_EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch+1}/{RESNET_EPOCHS} - Training Loss: {epoch_loss:.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        logging.info(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model
    try:
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        torch.save(model.state_dict(), RESNET_MODEL_PATH)
        logging.info(f"ResNet-50 model saved to {RESNET_MODEL_PATH}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def main():
    train_resnet50()

if __name__ == '__main__':
    main()
