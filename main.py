import argparse
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Import logic from the other files
from utils.baseline import run_training_pipeline
from utils.dataclass import SkinLesionDataset 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ConvNeXt Baseline")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default="../data/raw/")
    parser.add_argument('--csv_path', type=str, default="../data/raw/master_metadata_split.csv")
    parser.add_argument('--save_path', type=str, default="best_convnext_baseline.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    # Load Data
    baseline_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SkinLesionDataset(args.csv_path, args.data_dir, 'train', baseline_transforms)
    val_dataset = SkinLesionDataset(args.csv_path, args.data_dir, 'val', baseline_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 5)
    model = model.to(device)

    # Weights
    y_train = train_dataset.df['label'].map({'MEL': 0, 'NV': 1, 'BCC': 2, 'BKL': 3, 'AKIEC': 4}).values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Optimizer and Loss
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    run_training_pipeline(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, 'best_convnext_baseline.pth')