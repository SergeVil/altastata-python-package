import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 25 * 25, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_model(model_path='best_model.pth'):
    """Load the trained model"""
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Load and convert to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Resize
    image = F.resize(image, [100, 100])
    
    # Convert to tensor
    image_tensor = F.pil_to_tensor(image).float() / 255.0
    
    # Normalize
    image_tensor = F.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image

def predict(model, image_tensor):
    """Make prediction on a single image"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence

def main():
    # Load the model
    model = load_model()
    print("Model loaded successfully!")
    
    # Get list of test images
    import os
    test_images = [f for f in os.listdir('data/images') if f.endswith('.jpg')]
    test_images.sort()  # Sort for consistent order
    
    # Create a figure with subplots for all images
    n_images = len(test_images)
    n_cols = 5
    n_rows = (n_images + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(15, 3 * n_rows))
    
    # Process each image
    for idx, image_name in enumerate(test_images):
        image_path = os.path.join('data/images', image_name)
        
        try:
            # Preprocess image
            image_tensor, original_image = preprocess_image(image_path)
            
            # Make prediction
            predicted_class, confidence = predict(model, image_tensor)
            
            # Create subplot
            plt.subplot(n_rows, n_cols, idx + 1)
            plt.imshow(original_image)
            plt.title(f'{"Rectangle" if predicted_class else "Circle"}\n{confidence:.1%}')
            plt.axis('off')
            
            print(f"\nImage: {image_name}")
            print(f"Predicted class: {'Rectangle' if predicted_class else 'Circle'}")
            print(f"Confidence: {confidence:.2%}")
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 