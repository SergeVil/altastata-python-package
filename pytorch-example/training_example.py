import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from altastata import AltaStataPyTorchDataset
import altastata_config
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
    
    print("\n🚀 Starting PyTorch training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training', 
                         leave=False, ascii=True)
        
        for images, labels in train_pbar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar with current metrics
            current_loss = train_loss / (train_pbar.n + 1)
            current_acc = 100 * train_correct / train_total if train_total > 0 else 0
            train_pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.1f}%'})
        
        train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            # Create progress bar for validation
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation', 
                           leave=False, ascii=True)
            
            for images, labels in val_pbar:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar with current metrics
                current_loss = val_loss / (val_pbar.n + 1)
                current_acc = 100 * val_correct / val_total if val_total > 0 else 0
                val_pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.1f}%'})
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            # Only print significant improvements (>10% better) or milestones
            if best_val_loss == val_loss and (epoch + 1) % 25 == 0:  # Every 25 epochs
                print(f"\nEpoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.1f}% ✓")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.4f}")
                break
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n⚡ PyTorch training completed in {training_time:.1f} seconds!")
    
    # Load and save the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Save using the dataset
        model_save_path = 'pytorch_test/model/best_model.pth'
        print(f"\nSaving PyTorch model to AltaStata: {model_save_path}")
        train_dataset.save_model(best_model_state, model_save_path)
        
        # Model size info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel info:")
        print(f"Total parameters: {total_params:,}")
        print("Model saved successfully - ready for PyTorch inference! 🚀")
        print(f"Provenance file saved: {model_save_path}.provenance.txt with {len(train_dataset.file_paths)} file paths")
    
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
    train_dataset = AltaStataPyTorchDataset(
        "bob123_rsa",
        root_dir="pytorch_test/data/images",  # Fixed path to use correct location
        file_pattern="*.png",  # Updated pattern to match files directly
        transform=train_transform
    )
    
    val_dataset = AltaStataPyTorchDataset(
        "bob123_rsa",
        root_dir="pytorch_test/data/images",  # Fixed path to use correct location
        file_pattern="*.png",  # Updated pattern to match files directly
        transform=val_transform
    )
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Total files: {len(train_dataset)}")
    circle_count = sum(1 for path in train_dataset.file_paths if 'circle' in str(path))
    rectangle_count = sum(1 for path in train_dataset.file_paths if 'rectangle' in str(path))
    print(f"Circle images: {circle_count}")
    print(f"Rectangle images: {rectangle_count}\n")
    
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
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Model architecture: SimpleCNN with 3 conv blocks + classifier")
    
    train_model(model, train_loader, val_loader, train_dataset, criterion, optimizer, num_epochs=100, patience=10)

if __name__ == '__main__':
    main()