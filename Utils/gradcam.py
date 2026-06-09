import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, image_tensor, original_rgb_image, target_class_idx, save_name="gradcam.png"):
    """
    image_tensor: Normalized tensor of shape (1, 3, 224, 224) going into the model
    original_rgb_image: Unnormalized numpy array (224, 224, 3) with values 0-1 for visualization
    """
    model.eval()
    
    # ConvNeXt architectures typically hold their final convolutional blocks in features[-1]
    target_layers = [model.backbone.features[-1]]
    
    # Initialize standard Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Specify which class we want to see the heatmap for
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    # Generate the heatmap
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
    
    # Overlay heatmap on original image
    visualization = show_cam_on_image(original_rgb_image, grayscale_cam, use_rgb=True)
    
    # Plot side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_rgb_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(visualization)
    ax[1].set_title(f"Grad-CAM (Target Class: {target_class_idx})")
    ax[1].axis('off')
    
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM saved to {save_name}")