# AltaStata PyTorch Dataset

A custom PyTorch Dataset for reading files from local storage with support for multiple file types.

## Features

- Efficient file reading with automatic memory management
- Support for multiple file types:
  - Images (PNG, JPG, JPEG)
  - CSV files
  - NumPy arrays
- Customizable transforms for data preprocessing
- Automatic tensor conversion for PyTorch compatibility

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- PIL (Python Imaging Library)

## Quick Start

1. Clone the repository and install the package:
```bash
# Navigate to the package directory
cd altastata-pytorch-dataset

# Install the package in editable mode
pip install -e .
```

2. Navigate to examples directory:
```bash
cd examples
```

3. Clean up any existing data (optional):
```bash
rm -rf data best_model.pth
```

4. Generate sample data:
```bash
# Generate sample files (images, CSV, numpy)
python generate_sample_files.py
```
This will create:
- 5 circle images
- 5 rectangle images
- 5 CSV files
- 5 numpy files

5. Verify the generated data:
```bash
ls -R data/
```
You should see:
```
data/
├── csv/
│   └── sample_*.csv
├── images/
│   ├── circle_*.jpg
│   └── rectangle_*.jpg
└── numpy/
    └── sample_*.npy
```

6. Test dataset functionality:
```bash
python test_local_dataset.py
```
Expected output:
```
Image dataset size: 10
Image shape: torch.Size([3, 100, 100])
CSV dataset size: 5
CSV data shape: torch.Size([10, 5])
Numpy dataset size: 5
Numpy data shape: torch.Size([10, 5])
```

7. Train the model:
```bash
python training_example.py
```
This will:
- Train a CNN model to classify circles vs rectangles
- Save the best model as `best_model.pth`
- Show training progress and validation accuracy

8. Run inference:
```bash
python inference_example.py
```
This will:
- Load the trained model
- Process all test images
- Display predictions with confidence scores
- Show a grid of images with their classifications

## Usage

### Basic Usage
```python
from altastata_pytorch_dataset.altastata_pytorch_dataset import AltaStataPyTorchDataset
from torchvision import transforms

# Create image dataset with transforms
image_dataset = AltaStataPyTorchDataset(
    data_dir="path/to/images",
    pattern="*.jpg",
    transform=transforms.ToTensor()
)

# Create CSV dataset
csv_dataset = AltaStataPyTorchDataset(
    data_dir="path/to/csv",
    pattern="*.csv"
)

# Create numpy dataset
numpy_dataset = AltaStataPyTorchDataset(
    data_dir="path/to/numpy",
    pattern="*.npy"
)

# Use with DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Use in training loop
for data, target in dataloader:
    # Your training code here
    pass
```

## Configuration

### AltaStataPyTorchDataset Parameters:
- `data_dir`: Directory containing the data files
- `pattern`: Pattern to match files (e.g., "*.jpg" for images)
- `transform`: Optional transform to be applied to the data
- `target_transform`: Optional transform to be applied to the target

## File Type Support

### Images
- Supported formats: PNG, JPG, JPEG
- Automatically converted to RGB mode
- Can be used with torchvision transforms
- Default output: torch.Tensor with shape [C, H, W] and values normalized to [0, 1]

### CSV Files
- Automatically skips header row
- Converts numerical data to torch.FloatTensor
- Default output: 2D tensor with shape [rows, columns]

### NumPy Arrays
- Directly loads .npy files
- Converts to torch.FloatTensor
- Preserves original array shape

## Examples

See the `examples` directory for complete usage examples:

1. `generate_sample_files.py`: Creates sample files for testing
2. `test_local_dataset.py`: Demonstrates usage with different file types
3. `training_example.py`: Shows how to train a simple CNN model
4. `inference_example.py`: Demonstrates how to use the trained model for predictions

## Directory Structure
```
examples/
├── data/
│   ├── images/
│   │   ├── circle_*.jpg
│   │   └── rectangle_*.jpg
│   ├── csv/
│   │   └── sample_*.csv
│   └── numpy/
│       └── sample_*.npy
├── generate_sample_files.py
├── test_local_dataset.py
├── training_example.py
└── inference_example.py
```

## Training and Inference

The package includes examples for both training and inference:

### Training Example
The training example demonstrates how to:
1. Generate labeled training data (circles vs rectangles)
2. Create a dataset and data loaders
3. Train a simple CNN model for binary classification

To run the training example:
```bash
cd examples
python training_example.py
```

### Inference Example
The inference example shows how to:
1. Load a trained model from `best_model.pth`
2. Process new images for prediction
3. Make predictions with confidence scores
4. Display results with visualizations

To run the inference example:
```bash
cd examples
python inference_example.py
```

## Troubleshooting

If you encounter any issues:
1. Make sure all required packages are installed
2. Check that sample files were generated successfully
3. Verify file paths and permissions
4. Ensure your data files match the expected formats:
   - Images should be readable by PIL
   - CSV files should contain numeric data
   - NumPy files should be valid .npy format

## License

[Your License Here] 