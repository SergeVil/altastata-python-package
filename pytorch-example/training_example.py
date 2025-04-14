import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from altastata import AltaStataPyTorch
from altastata_config import altastata_functions
import numpy as np
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a global AltaStataPyTorch instance
altastata = AltaStataPyTorch(altastata_functions)

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

def train_model(model, train_loader, val_loader, train_dataset, criterion, optimizer, num_epochs=100, patience=10):
    """Train the model with early stopping."""
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            print("  Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
    
    # Load and save the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Save using the dataset
        train_dataset.save_model(best_model_state, 'models/best_model.pth')
    
    print("\nTraining completed!")

def main():
    # Create transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create validation transform without augmentation
    val_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets using AltaStataPyTorch
    train_dataset = altastata.create_dataset(
        root_dir="data/images",  # Fixed path to use correct location
        file_pattern="*.png",  # Updated pattern to match files directly
        transform=train_transform
    )
    
    val_dataset = altastata.create_dataset(
        root_dir="data/images",  # Fixed path to use correct location
        file_pattern="*.png",  # Updated pattern to match files directly
        transform=val_transform
    )
    
    # Print all available files
    print("\nAll file paths in dataset:")
    for path in train_dataset.file_paths:
        if isinstance(path, Path):
            print(f"  {path.name}")
        else:
            # For cloud storage, print the full path
            print(f"  {path}")
    print()
    
    # Create data indices for training and validation splits
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
        sampler=train_sampler,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        sampler=val_sampler,
        num_workers=0
    )
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    train_model(model, train_loader, val_loader, train_dataset, criterion, optimizer, num_epochs=100, patience=10)

if __name__ == '__main__':
    main()