import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from efficient_kan import KAN 

from Utils.dataclass import SkinLesionDataset

# KCN framework
class ConvNeXtKAN(nn.Module):
    def __init__(self, num_classes=5, kan_hidden_dim=32):
        super(ConvNeXtKAN, self).__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = KAN([in_features, kan_hidden_dim, num_classes])

        # self.classifier_head = nn.Sequential(
        #     nn.Dropout(p=dropout_p),
        #     KAN([in_features, kan_hidden_dim, num_classes])
        # )

        # self.backbone.classifier[2] = self.classifier_head

    def forward(self, x):
        return self.backbone(x)

    def get_regularization_loss(self):
        return self.backbone.classifier[2].regularization_loss()
        # return self.backbone.classifier[2][1].regularization_loss()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg_weight', type=float, default=1e-3, help="Weight of KAN L1 penalty")
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout probability")
    parser.add_argument('--data_dir', type=str, default="../data/raw/")
    parser.add_argument('--csv_path', type=str, default="../data/raw/master_metadata_split.csv")
    parser.add_argument('--save_path', type=str, default="best_KCN_baseline.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting KCN Training on {device} ---")

    train_transforms = transforms.Compose([
        transforms.Resize((236, 236)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SkinLesionDataset(args.csv_path, args.data_dir, 'train', train_transforms)
    val_dataset = SkinLesionDataset(args.csv_path, args.data_dir, 'val', val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model & Optimization
    # model = ConvNeXtKAN(num_classes=5, kan_hidden_dim=32, dropout_p=args.dropout).to(device)
    model = ConvNeXtKAN(num_classes=5, kan_hidden_dim=32).to(device)

    # Balance weights
    y_train = train_dataset.df['label'].map({'MEL': 0, 'NV': 1, 'BCC': 2, 'BKL': 3, 'AKIEC': 4}).values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = 0.0
    best_val_loss = float('inf')

    loss_save_path = args.save_path.replace('.pth', '_best_loss.pth')
    f1_save_path = args.save_path.replace('.pth', '_best_f1.pth')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)

            ce_loss = criterion(outputs, labels)
            reg_loss = model.get_regularization_loss()
            loss = ce_loss + (args.reg_weight * reg_loss)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            train_total += labels.size(0)

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                ce_loss = criterion(outputs, labels)
                reg_loss = model.get_regularization_loss()
                loss = ce_loss + (args.reg_weight * reg_loss)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_acc = accuracy_score(val_labels, val_preds)
        train_acc = train_correct / train_total

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        # print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val F1: {val_f1:.4f}")

        # Strict Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), loss_save_path)
            print(f"  --> Generalization Checkpoint Saved! (Best Loss:{best_val_loss:.4f} | F1: {val_loss/len(val_loader):.4f})")
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f1_save_path)
            print(f"  --> Accuracy Checkpoint Saved! (Best F1: {best_val_f1:.4f} | Val Loss: {avg_val_loss:.4f})")

if __name__ == "__main__":
    main()