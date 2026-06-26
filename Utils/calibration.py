import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def plot_calibration_curve(y_true, y_logits, num_bins=10, save_name="calibration.png"):
    print("Calculating Confidence Calibration...")
    
    # Convert logits to probabilities
    probabilities = F.softmax(torch.tensor(y_logits), dim=1).numpy()
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = predictions == y_true
    
    # Setup bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bins, right=True)
    
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    ece = 0.0
    
    for b in range(1, num_bins + 1):
        mask = bin_indices == b
        if np.any(mask):
            bin_accuracies[b-1] = np.mean(accuracies[mask])
            bin_confidences[b-1] = np.mean(confidences[mask])
            bin_counts[b-1] = np.sum(mask)
            
            # ECE
            prob_in_bin = bin_counts[b-1] / len(confidences)
            ece += prob_in_bin * np.abs(bin_accuracies[b-1] - bin_confidences[b-1])
            
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

    valid_bins = bin_counts > 0
    plt.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], marker='o', linewidth=2, label=f'Model (ECE: {ece:.4f})')
    
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Positives (Accuracy)')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration diagram saved to {save_name}")