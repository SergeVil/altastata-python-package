import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from altastata import AltaStataPyTorchDataset
import altastata_config
import os
from pathlib import Path
import io

# Configure matplotlib for Jupyter notebook if running in one
try:
    # Check if we're running in a Jupyter notebook
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
        get_ipython().run_line_magic('matplotlib', 'inline')
    elif shell == 'TerminalInteractiveShell':  # IPython terminal
        pass
except:
    pass  # Regular Python interpreter

# Define the same model architecture as in training
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 12 * 12, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_model(model_path):
    """Load the trained model."""
    model = SimpleCNN()
    # Create dataset with file pattern to filter only .pth files (excludes provenance files)
    model_dataset = AltaStataPyTorchDataset(
        "bob123_rsa",
        root_dir="pytorch_test/model",
        file_pattern="*.pth",  # Pattern matches all .pth files, excludes .provenance.txt
        require_files=False
    )
    
    # Load using the resolved file path (pattern already filtered to .pth files)
    model.load_state_dict(model_dataset.load_model(model_dataset.file_paths[0]))
    model.eval()
    return model

def display_images(images, predictions, confidences):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 6))
    if len(images) == 1:
        axes = [axes]
    
    for ax, image, pred, conf in zip(axes, images, predictions, confidences):
        ax.imshow(np.array(image))
        ax.set_title(f'Predicted: {"Circle" if pred == 1 else "Rectangle"}\nConfidence: {conf:.2f}%')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Define transforms for inference (no augmentation needed)
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the trained model
    model = load_model("pytorch_test/model/best_model.pth")
    print("Model loaded successfully!")
    print("=" * 50)
    
    # Create dataset for inference
    test_dataset = AltaStataPyTorchDataset(
        "bob123_rsa",
        root_dir="pytorch_test/data/images",  # Changed to use root data directory
        file_pattern="*.png",  # Updated pattern to match subdirectory
        transform=transform
    )
    
    # Print available files
    print("\nAvailable files in dataset:")
    for path in test_dataset.file_paths:
        if isinstance(path, Path):
            print(f"  {path.name}")
        else:
            # For cloud storage, print the full path
            print(f"  {path}")
    print()
    
    # Test on specific images
    test_indices = [0, 5]  # circle_0.png and rectangle_0.png
    
    # Collect results for batch display
    images = []
    predictions = []
    confidences = []
    
    for idx in test_indices:
        # Get data and label from dataset
        image_tensor, true_label = test_dataset[idx]
        
        # Get the original image
        if isinstance(test_dataset.file_paths[idx], Path):
            original_image = Image.open(test_dataset.file_paths[idx]).convert('RGB')
        else:
            # For cloud storage, use _read_file
            image_bytes = test_dataset._read_file(test_dataset.file_paths[idx])
            original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        # Store results
        images.append(original_image)
        predictions.append(predicted_class)
        confidences.append(confidence)
        
        # Print results
        file_path = test_dataset.file_paths[idx]
        if isinstance(file_path, Path):
            print(f"\nImage: {file_path.name}")
        else:
            # For cloud storage, get the base filename
            print(f"\nImage: {os.path.basename(file_path.split('✹')[0])}")
        print(f"True Label: {'Circle' if true_label == 1 else 'Rectangle'}")
        print(f"Predicted: {'Circle' if predicted_class == 1 else 'Rectangle'}")
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 50)
    
    # Display all images together
    display_images(images, predictions, confidences)

if __name__ == "__main__":
    main() 