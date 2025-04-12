import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from altastata import AltaStataPyTorchDataset
import numpy as np

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Print first batch info
        for i, (images, labels) in enumerate(train_loader):
            if i == 0:
                print("\nFirst training batch:")
                print(f"Images shape: {images.shape}")
                print(f"Labels shape: {labels.shape}\n")
            break
            
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch + 1}:")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
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
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), 'best_model.pth')
    
    print("\nTraining completed!")

def main():
    # Create transforms with data augmentation
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = AltaStataPyTorchDataset(
        root_dir="data/images",
        file_pattern="*.png",
        transform=transform
    )
    
    val_dataset = AltaStataPyTorchDataset(
        root_dir="data/images",
        file_pattern="*.png",
        transform=transform
    )
    
    print("\nAll file paths in dataset:")
    for path in train_dataset.file_paths:
        print(path.name)
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
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10)

if __name__ == '__main__':
    main()