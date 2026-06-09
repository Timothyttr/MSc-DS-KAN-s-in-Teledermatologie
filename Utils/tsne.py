import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    Bypasses the classification head to return raw ConvNeXt feature embeddings.
    """
    model.eval()
    features_list = []
    
    for images, _ in dataloader:
        images = images.to(device)
        
        # Pass through ConvNeXt feature extractor
        x = model.backbone.features(images)
        x = model.backbone.avgpool(x)
        x = torch.flatten(x, 1) # This gives us the [Batch, 768] vector
        
        features_list.append(x.cpu().numpy())
        
    return np.vstack(features_list)

def plot_domain_tsne(ham_features, pad_features, save_name="domain_tsne.png"):
    print("Running t-SNE dimensionality reduction (This will take a minute)...")
    
    # Combine features and create labels for the domains
    all_features = np.vstack([ham_features, pad_features])
    domain_labels = ['HAM10000 (Clinical)'] * len(ham_features) + ['PAD-UFES-20 (Smartphone)'] * len(pad_features)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(all_features)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:, 0], 
        y=tsne_results[:, 1],
        hue=domain_labels,
        palette=['#1f77b4', '#ff7f0e'],
        alpha=0.6,
        s=15
    )
    plt.title('t-SNE Feature Space: Domain Shift')
    plt.legend(title='Domain')
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {save_name}")