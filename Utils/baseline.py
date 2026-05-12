import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return total_loss / total, 100. * correct / total

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    val_acc = 100. * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return total_loss / total, val_acc, val_f1

def run_training_pipeline(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path):
    print(f"\n--- Starting ConvNeXt Baseline Training on {device} ---")
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.0f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  --> Weights saved! (New Best Val F1: {best_val_f1:.4f})")