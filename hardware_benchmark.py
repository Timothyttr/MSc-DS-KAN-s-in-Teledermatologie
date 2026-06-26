import time
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

from efficient_kan import KAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on: {device}")

def get_baseline(num_classes=5):
    model = convnext_tiny(weights=None) 
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

class ConvNeXtKAN(nn.Module):
    def __init__(self, num_classes=5, kan_hidden_dim=32):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)
        self.backbone.classifier[2] = KAN([self.backbone.classifier[2].in_features, kan_hidden_dim, num_classes])
        
    def forward(self, x):
        return self.backbone(x)

def load_fixed_weights(model, path, device):
    raw_dict = torch.load(path, map_location=device)
    fixed_dict = {}
    for key, value in raw_dict.items():
        # Baseline key fix
        key = key.replace('classifier.2.1.', 'classifier.2.')
        # KCN key fix
        key = key.replace('classifier_head.1.', 'backbone.classifier.2.')
        fixed_dict[key] = value
        
    model.load_state_dict(fixed_dict, strict=True)
    return model


def benchmark_model(model, model_name, input_size=(1, 3, 224, 224), iterations=100):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(10): 
            _ = model(dummy_input)
            
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations): 
            _ = model(dummy_input)
            
    if torch.cuda.is_available(): 
        torch.cuda.synchronize()
        
    total_time = time.time() - start_time
    fps = iterations / total_time
    
    print(f"{model_name:<22} | Params: {total_params:>10,} | FPS: {fps:.2f} Images/Sec")


print("\nLoading Models...")
try:
    base_model = get_baseline().to(device)
    base_model = load_fixed_weights(base_model, "VAULT/BASELINE/Balanced/Paths/baseline_exp1_std_reg_best_loss.pth", device)
    base_model.eval()

    kcn_model = ConvNeXtKAN().to(device)
    kcn_model = load_fixed_weights(kcn_model, "VAULT/KCN/Balanced/Paths/kcn_heavy_spline_best_loss.pth", device)
    kcn_model.eval()

    print("\n========================================")
    print("         GPU HARDWARE BENCHMARK         ")
    print("========================================")
    benchmark_model(base_model, "Baseline Regularized")
    benchmark_model(kcn_model, "KCN Heavy Spline")
    print("========================================\n")
except Exception as e:
    print(f"Error loading models: {e}")
