# PyTorch Dataset Example

Examples for using the `altastata` package with `AltaStataPyTorch` - a high-level class that provides a unified interface for PyTorch dataset operations and AltaStata functionality.

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

## Features

### File Content Cache
The dataset includes an intelligent file content cache that:
- Automatically caches files up to 16MB in size
- Maintains a total cache size limit of 1GB
- Removes files from cache when they are modified
- Provides detailed logging of cache operations
- Improves performance for frequently accessed files

### Multi-Process Support
The dataset is designed to work efficiently with PyTorch's DataLoader:
- Supports multiple worker processes
- Properly handles file access across processes
- Includes process-specific logging
- Maintains cache consistency across processes

## Example Descriptions

### Basic Dataset Usage
The examples demonstrate how to use the dataset with various file types:

```python
# Import the required classes from the altastata package
from altastata import AltaStataPyTorch, AltaStataFunctions
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

# Create a global AltaStataPyTorch instance
altastata = AltaStataPyTorch(AltaStataFunctions(...))

# Create dataset with transforms
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset using the global instance
dataset = altastata.create_dataset(
    root_dir="data/images",
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
- Efficient multi-process data loading

### Inference Example
The inference example shows how to:
- Load a trained model
- Preprocess new images
- Make predictions
- Display results with confidence scores
- Visualize predictions
- Use the file content cache for faster inference

## Project Structure
```
pytorch-example/              # Examples using the altastata package
    test_dataset.py           # Basic dataset tests
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
    __init__.py               # Exports classes including AltaStataPyTorch
    altastata_functions.py    # Original altastata functionality
    altastata_pytorch_dataset.py # Implementation of the dataset class
```

## License

MIT License 