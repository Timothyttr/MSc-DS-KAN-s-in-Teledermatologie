import torch
import pandas as pd
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score
from torchvision import models, transforms

from Utils.dataclass import SkinLesionDataset

def get_baseline(num_classes=5, dropout_p=0.5):
    #convnext_tiny
    model = models.convnext_tiny(weights=None) 
    in_features = model.classifier[2].in_features
    # model.classifier[2] = nn.Linear(model.classifier[2].in_features, 5)

    model.classifier[2] = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout_p),
        torch.nn.Linear(in_features, num_classes)
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="Run blind evaluation on test set.")
    parser.add_argument('--weights', type=str, required=True, help="the .pth weights file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Inference on {device} ---")
    print(f"Evaluating Model Weights: {args.weights}")

    model = get_baseline(num_classes=5)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval() 

    csv_path = "/scratch-shared/ttoonen/data/raw/master_metadata_split.csv"
    data_dir = "/scratch-shared/ttoonen/data/raw/"
    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'test']
    print(f"Loaded {len(test_df)} out-of-distribution smartphone images.")

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = SkinLesionDataset(csv_path, data_dir, 'test', test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []
    all_probs = []

    print("Beginning baseline eval.")
    with torch.no_grad():
        for images, labels in test_loader: 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Extract probabilities and predictions
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    target_names = ['MEL', 'NV', 'BCC', 'BKL', 'AKIEC']
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    print("\n=== FINAL TEST METRICS (Baseline on PAD-UFES-20) ===")
    print(f"Macro F1 Score:      {macro_f1:.4f}")
    print(f"Weighted F1 Score:   {weighted_f1:.4f}")
    print(f"Balanced Accuracy:   {bal_acc:.4f}")
    print(f"Macro ROC-AUC:       {macro_roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

if __name__ == "__main__":
    main()