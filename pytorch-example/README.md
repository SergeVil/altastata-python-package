# PyTorch Dataset Example

Examples for using the `altastata` package with AltaStataPyTorchDataset - a flexible PyTorch Dataset implementation that supports working with various file types including images, CSV files, and NumPy arrays.

## Package Installation

The implementation has been moved to the main `altastata` package. To install it:

```bash
# Navigate to the altastata package directory
cd altastata-python-package

# Install in development mode
pip install -e .
```

### Dependencies

The package requires the following dependencies:
- torch
- torchvision
- numpy
- pandas
- Pillow (Python Imaging Library)

## Running Examples

After ensuring the `altastata` package is installed, you can run the examples:

```bash
cd pytorch-example
python generate_sample_files.py
python test_dataset.py
python training_example.py
python inference_example.py
```

## Example Descriptions

### Basic Dataset Usage
The examples demonstrate how to use the dataset with various file types:

```python
# Import the dataset class from the altastata package
from altastata import AltaStataPyTorchDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

# Create dataset with transforms
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = AltaStataPyTorchDataset(
    root_dir="path/to/data",
    file_pattern="*.jpg",  # or *.npy, *.csv
    transform=transform
)

# Use with DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Use in training loop
for data, labels in dataloader:
    # data is a tensor of shape [batch_size, channels, height, width] for images
    # labels are automatically generated based on filenames (1 for 'circle', 0 for others)
    pass
```

### Training Example
The package includes a training example that demonstrates:
- Loading and preprocessing images
- Training a CNN model
- Model validation and saving
- Binary classification (circles vs rectangles)
- Data augmentation during training
- Early stopping and model checkpointing

### Inference Example
The inference example shows how to:
- Load a trained model
- Preprocess new images
- Make predictions
- Display results with confidence scores
- Visualize predictions

## Project Structure
```
pytorch-example/              # Examples using the altastata package
    test_dataset.py           # Basic dataset tests
    test_local_dataset.py     # Local filesystem tests
    training_example.py       # CNN training example
    inference_example.py      # Model inference example
    generate_sample_files.py  # Creates sample data
    data/                     # Directory for sample data
        images/               # Sample images
        csv/                  # CSV files
        numpy/                # NumPy arrays
        models/              # Saved model checkpoints
    README.md                 # This documentation file

altastata/                    # Main package (separate location)
    __init__.py               # Exports classes including AltaStataPyTorchDataset
    altastata_functions.py    # Original altastata functionality
    altastata_pytorch_dataset.py # Implementation of the dataset class
```

## License

MIT License 