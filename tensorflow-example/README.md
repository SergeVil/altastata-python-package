# TensorFlow Dataset Example

Examples for using the `altastata` package with TensorFlow - a high-level interface for TensorFlow dataset operations and AltaStata functionality.

## Package Installation

The implementation is part of the main `altastata` package. To install it:

```bash
# Navigate to the altastata package directory
cd altastata-python-package

# Install in development mode
pip install -e .
```

### Dependencies

The package requires the following dependencies:
- tensorflow
- numpy
- Pillow (Python Imaging Library)
- matplotlib (for visualization)

## Running Examples

After ensuring the `altastata` package is installed, you can run the examples:

```bash
cd tensorflow-example
python test_dataset.py
python training_example.py
python inference_example.py
```

## Features

### Direct AltaStata Integration
The dataset integrates directly with AltaStata file storage:
- Creates models and saves them to AltaStata storage
- Loads models from AltaStata storage
- Reads and processes images from AltaStata
- Works with both local files and AltaStata cloud storage

### File Content Cache
The dataset includes an intelligent file content cache that:
- Automatically caches files up to 16MB in size
- Maintains a total cache size limit of 1GB
- Provides detailed logging of cache operations
- Improves performance for frequently accessed files

### Multi-Process Support
The dataset is designed to work efficiently with TensorFlow's data pipeline:
- Supports parallel data loading
- Properly handles file access across processes
- Includes process-specific logging
- Maintains cache consistency across processes

## Example Descriptions

### Basic Dataset Usage
The examples demonstrate how to use the dataset with various file types:

```python
# Import the dataset class from the altastata package
from altastata import AltaStataTensorFlowDataset
import tensorflow as tf


# Create dataset with preprocessing
def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [96, 96])
    return image, label


# Create dataset using account ID
dataset = AltaStataTensorFlowDataset(
    "bob123_rsa",  # Account ID for AltaStata
    root_dir="tensorflow_test/data/images",
    file_pattern="*.png",
    preprocess_fn=preprocess_image
)

# Use with TensorFlow's data pipeline
dataset = dataset.batch(8).prefetch(tf.data.AUTOTUNE)

# Use in training loop
for data, labels in dataset:
    # data is a tensor of shape [batch_size, 96, 96, 3] for images
    # labels are automatically generated based on filenames (1 for 'circle', 0 for 'rectangle')
    pass
```

### Training Example
The package includes a training example that demonstrates:
- Loading and preprocessing images from AltaStata
- Training a CNN model with edge detection
- Proper shape feature detection
- Model validation and saving to AltaStata
- Binary classification (circles vs rectangles)
- Data augmentation during training
- Early stopping and learning rate reduction
- Checkpoint saving to AltaStata

### Inference Example
The inference example shows how to:
- Load a trained model from AltaStata
- Preprocess new images
- Make predictions
- Display results with confidence scores
- Visualize predictions
- Calculate and report overall accuracy

## Project Structure
```
tensorflow-example/           # Examples using the altastata package
    test_dataset.py           # Basic dataset tests
    training_example.py       # CNN training example
    inference_example.py      # Model inference example
    data_tensorflow/          # Directory for sample data
        images/               # Sample images
        models/               # Saved model checkpoints
        models/checkpoints/   # Training checkpoints
    README.md                 # This documentation file

altastata/                    # Main package (separate location)
    __init__.py               # Exports classes including AltaStataTensorFlowDataset
    altastata_functions.py    # Original altastata functionality
    altastata_tensorflow_dataset.py # Implementation of the dataset class
```

## License

MIT License 